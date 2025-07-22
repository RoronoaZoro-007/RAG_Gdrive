#!/bin/bash

# Setup script for Internal Onboarding AI
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Internal Onboarding AI environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.13 or later."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "📦 Using Python version: $PYTHON_VERSION"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies from requirements_fixed.txt..."
pip install -r requirements_fixed.txt

# Download spaCy model
echo "🤖 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Test the installation
echo "🧪 Testing installation..."
python -c "import streamlit_app_drive_opensearch; print('✅ All dependencies installed successfully!')"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application, run:"
echo "  streamlit run streamlit_app_drive_opensearch.py"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate" 