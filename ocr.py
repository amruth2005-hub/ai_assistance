# Enhanced OCR module with preprocessing pipeline
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
from pdf2image import convert_from_path

class EnhancedOCR:
    def __init__(self):
        # Configuration options
        self.config = {
            'lang': 'eng',
            'psm': 6,  # Assume single block of text
            'oem': 3   # Default OCR engine mode
        }
    
    def preprocess_image(self, image):
        """Apply advanced preprocessing to improve OCR accuracy"""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Check if the background is dark
        corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
        avg_corner = sum(corners) / 4
        
        if avg_corner < 127:
            gray = cv2.bitwise_not(gray)
        
        # Noise removal with bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 31, 10)
        
        # Dilation to make characters more robust
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    
    def detect_lines(self, image):
        """Detect horizontal lines for answer separation"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        return lines
    
    def process_image(self, image_path, detect_questions=False):
        """Process image and extract text with optional question detection"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image at path: {image_path}")
            
            # Preprocess image
            processed = self.preprocess_image(img)
            
            # Custom configuration based on parameters
            config = f"-l {self.config['lang']} --psm {self.config['psm']} --oem {self.config['oem']}"
            
            # Extract text
            text = pytesseract.image_to_string(processed, config=config)
            
            # If question detection is enabled
            if detect_questions:
                # Use structural analysis to detect question numbers
                question_bounds = self.detect_question_regions(processed)
                questions = {}
                
                for idx, (x, y, w, h) in enumerate(question_bounds):
                    question_img = processed[y:y+h, x:x+w]
                    question_text = pytesseract.image_to_string(question_img, config=config)
                    questions[idx+1] = question_text
                
                return {"full_text": text, "questions": questions}
            
            return {"full_text": text}
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {"error": str(e)}
    
    def detect_question_regions(self, image):
        """Detect regions containing questions based on formatting patterns"""
        # This is a placeholder for question detection logic
        # In a real implementation, this would analyze the layout and detect question areas
        # based on numbering, indentation, or other visual cues
        
        # For now, we'll return a simple placeholder
        h, w = image.shape
        return [(0, 0, w, h//4), (0, h//4, w, h//4), (0, h//2, w, h//2)]
    
    def pdf_to_text(self, pdf_path, detect_questions=False):
        """Process multi-page PDF documents"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir)
                full_text = ""
                all_questions = {}
                
                for i, image in enumerate(images):
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    image.save(image_path, "PNG")
                    
                    result = self.process_image(image_path, detect_questions)
                    
                    if "error" not in result:
                        full_text += result["full_text"] + f"\n--- Page {i+1} End ---\n"
                        
                        if detect_questions and "questions" in result:
                            # Offset question numbers by page
                            page_questions = {f"{i+1}.{q_num}": text 
                                           for q_num, text in result["questions"].items()}
                            all_questions.update(page_questions)
                    else:
                        full_text += f"[Error processing page {i+1}: {result['error']}]\n"
                
                return {
                    "full_text": full_text,
                    "questions": all_questions if detect_questions else {}
                }
                
        except Exception as e:
            print(f"An error occurred processing PDF: {e}")
            return {"error": str(e)}
