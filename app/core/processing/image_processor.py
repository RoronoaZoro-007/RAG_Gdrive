"""
Image processing for PDF documents.
Handles image extraction and description generation using Gemini Vision.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io
import fitz  # PyMuPDF
import uuid
import base64
from config.settings import settings
from config.constants import GEMINI_MODEL
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ImageProcessor:
    """Handles image processing and description generation."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.model = self.init_gemini(settings.GOOGLE_API_KEY)
        
        # Thread-local storage for model
        self._local = threading.local()
    
    def _get_model(self):
        """Get thread-local model instance."""
        if not hasattr(self._local, 'model'):
            self._local.model = self.init_gemini(settings.GOOGLE_API_KEY)
        return self._local.model
    
    def init_gemini(self, api_key: str) -> ChatGoogleGenerativeAI:
        """
        Initialize Gemini API client for image processing.
        
        Args:
            api_key (str): Google API key for Gemini access
            
        Returns:
            ChatGoogleGenerativeAI: Configured Gemini model instance
        """
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.1
        )
    
    def describe_image(self, image_data, prompt: str = "What's in this image? Provide full detail as possible.") -> str:
        """
        Generate detailed description of an image using Gemini Vision.
        
        Args:
            image_data: Image data (bytes or PIL Image)
            prompt (str): Custom prompt for image analysis
            
        Returns:
            str: Detailed description of the image content
        """
        try:
            # Convert bytes to PIL Image if needed
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            # Convert PIL image to base64 for LangChain
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            image_url = f"data:image/png;base64,{img_str}"
            
            # Create message with image and text
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": image_url},
                ]
            )
            
            # Generate description
            response = self._get_model().invoke([message])
            return response.content
            
        except Exception as e:
            return None
    
    def extract_images_from_pdf(self, pdf_path: str) -> list:
        """
        Extract all images from a PDF file using PyMuPDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries containing image data and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            images.append({
                                "page": page_num + 1,
                                "image_data": img_data,
                                "image_id": f"page_{page_num+1}_img_{img_index}_{uuid.uuid4().hex[:8]}"
                            })
                        
                        pix = None  # Free memory
                    except Exception as e:
                        continue
            
            doc.close()
            return images
        except Exception as e:
            return []
    
    def _process_single_image(self, img):
        """Process a single image (for parallel processing)."""
        try:
            print(f"🔄 [PARALLEL] Processing image {img['image_id']} from page {img['page']}")
            description = self.describe_image(img["image_data"])
            if description:
                result = {
                    "page": img["page"],
                    "image_id": img["image_id"],
                    "description": description
                }
                print(f"✅ [PARALLEL] Successfully processed image {img['image_id']}")
                return result
            else:
                print(f"❌ [PARALLEL] Failed to process image {img['image_id']}")
                return None
        except Exception as e:
            print(f"❌ [PARALLEL] Error processing image {img.get('image_id', 'unknown')}: {str(e)}")
            return None

    def process_pdf_images(self, pdf_path: str) -> list:
        """
        Process all images in a PDF and generate descriptions for each using parallel processing.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries with image descriptions and metadata
        """
        images = self.extract_images_from_pdf(pdf_path)
        results = []
        
        if not images:
            print("📄 No images found in PDF")
            return results
        
        print(f"🚀 [PARALLEL] Starting parallel image processing for {len(images)} images...")
        
        # Use ThreadPoolExecutor for parallel image processing
        with ThreadPoolExecutor(max_workers=min(4, len(images))) as executor:
            # Submit all image processing tasks
            image_futures = {
                executor.submit(self._process_single_image, img): img 
                for img in images
            }
            
            # Collect results as they complete
            for future in as_completed(image_futures):
                img = image_futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"❌ [PARALLEL] Exception in image processing: {str(e)}")
        
        print(f"✅ [PARALLEL] Image processing completed: {len(results)} successful out of {len(images)} images")
        return results 