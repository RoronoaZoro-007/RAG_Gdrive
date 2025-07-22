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

class ImageProcessor:
    """Handles image processing and description generation."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.model = self.init_gemini(settings.GOOGLE_API_KEY)
    
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
            response = self.model.invoke([message])
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
    
    def process_pdf_images(self, pdf_path: str) -> list:
        """
        Process all images in a PDF and generate descriptions for each.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries with image descriptions and metadata
        """
        images = self.extract_images_from_pdf(pdf_path)
        results = []
        
        for i, img in enumerate(images):
            description = self.describe_image(img["image_data"])
            if description:
                results.append({
                    "page": img["page"],
                    "image_id": img["image_id"],
                    "description": description
                })
        
        return results 