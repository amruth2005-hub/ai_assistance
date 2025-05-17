# AI-Assisted Grading and Evaluation System

## Project Overview

This project implements a comprehensive AI-assisted grading and evaluation system according to the IEEE project report. It transforms the basic OCR script into a full-featured web application with an intuitive dashboard interface that can intelligently process handwritten exam papers.

## Core Features

1. **Enhanced OCR Module**:
   - Improved image preprocessing pipeline for better text recognition
   - Question detection and segmentation
   - Support for multiple document formats (PDF, image files)

2. **Answer Mapping Module**:
   - Intelligent association of extracted text with corresponding questions
   - Multiple detection methods for different document structures
   - Manual mapping correction capabilities

3. **Grading Module**:
   - AI-powered semantic matching against reference answers
   - Bloom's Taxonomy cognitive level classification
   - Configurable grading parameters and feedback generation

4. **Question Paper Generator**:
   - Structured question paper creation based on blueprints
   - Bloom's Taxonomy cognitive level distribution
   - Model answer generation for grading reference

5. **Analytics Dashboard**:
   - Performance visualization and statistical analysis
   - Cognitive level breakdown
   - Export functionality for results and reports

6. **Batch Processing**:
   - Process multiple papers simultaneously
   - Class-wide performance analysis
   - Bulk export options

## System Architecture

The system follows a modular architecture with these key components:

1. **Core Processing Modules**:
   - `enhanced_ocr.py` - Handles document scanning and text extraction
   - `answer_mapping.py` - Maps extracted text to questions
   - `grading.py` - Evaluates answers using AI algorithms
   - `question_generator.py` - Creates structured question papers

2. **Web Application**:
   - `app.py` - Flask server with route definitions
   - Templates - HTML/CSS/JS for the web interface
   - Static assets - CSS, JavaScript, and images

## Installation Instructions

### Prerequisites
```
Python 3.8+
Tesseract OCR (installed and added to PATH)
Poppler (installed bin to PATH)
Virtual environment (recommended)
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-grading-system.git
cd ai-grading-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install external dependencies:
   - Tesseract OCR: [Instructions](https://github.com/tesseract-ocr/tesseract)
   - Poppler: [Instructions](https://poppler.freedesktop.org/)

### Running the Application

Start the Flask development server:
```bash
python app.py
```

The application will be available at http://localhost:5000/

## Project Structure

```
ai-grading-system/
├── app.py                 # Main Flask application
├── enhanced_ocr.py        # Enhanced OCR module
├── answer_mapping.py      # Answer mapping module
├── grading.py             # Grading system
├── question_generator.py  # Question paper generator
├── requirements.txt       # Project dependencies
├── uploads/               # Directory for uploaded files
├── static/                # Static assets (CSS, JS)
│   ├── css/               # CSS stylesheets
│   ├── js/                # JavaScript files
│   └── images/            # Image assets
└── templates/             # HTML templates
    ├── base.html          # Base template
    ├── index.html         # Dashboard
    ├── ocr.html           # OCR scanning
    ├── mapping.html       # Answer mapping
    ├── grading.html       # Grading interface
    ├── question_generator.html  # Question generation
    ├── analytics.html     # Analytics dashboard
    └── batch_processing.html  # Batch processing
```

## Usage Guide

### 1. Dashboard
- View system status and quick actions
- Access all system modules

### 2. OCR Scanning
- Upload handwritten exam papers (PDF or images)
- Configure OCR parameters for optimal results
- Review extracted text before proceeding

### 3. Answer Mapping
- Map extracted text to corresponding questions
- Use auto-mapping or manually adjust mappings
- Validate mapping accuracy

### 4. Generate Question Papers
- Create structured question papers
- Configure cognitive level distribution
- Generate model answers for grading reference

### 5. Grading
- Evaluate mapped answers against references
- Review and adjust AI-generated scores if needed
- Add manual feedback

### 6. Analytics
- View detailed performance metrics
- Analyze cognitive level breakdown
- Export results in various formats

### 7. Batch Processing
- Process multiple submissions at once
- View class-wide performance statistics
- Export batch results

## Technology Stack

- **Backend**: Python, Flask
- **OCR Engine**: Tesseract, OpenCV
- **Natural Language Processing**: NLTK, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Visualization**: Matplotlib, Charts.js
- **Document Processing**: pdf2image, Pillow

## Future Enhancements

1. **Machine Learning Improvements**:
   - Train custom models for specific subject domains
   - Continuous learning from grader feedback

2. **Integration Capabilities**:
   - LMS integration (Canvas, Moodle, etc.)
   - Student portal access

3. **Advanced Analytics**:
   - Longitudinal student performance tracking
   - Question effectiveness analysis

4. **Mobile Support**:
   - Mobile app for on-the-go grading
   - Mobile-optimized scanning

## Contributors

- Kirti Akshaya Rangu
- Nikhil Emmanuel
- Rupal Sharma
- Kovuru Hemamruth
