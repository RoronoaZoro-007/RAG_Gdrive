#!/usr/bin/env python3
"""
Main entry point for the SheetsLoader application.
Run this script from the root directory to start the application.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the main function
from app.ui import main

if __name__ == "__main__":
    main() 