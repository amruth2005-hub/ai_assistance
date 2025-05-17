# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import uuid
import json
from werkzeug.utils import secure_filename
import tempfile
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# Import our custom modules
from enhanced_ocr import EnhancedOCR
from answer_mapping import AnswerMapper
from grading import GradingSystem
from question_generator import QuestionPaperGenerator

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize our components
ocr_engine = EnhancedOCR()
answer_mapper = AnswerMapper()
grading_system = GradingSystem()
question_generator = QuestionPaperGenerator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + filename)
            file.save(file_path)
            
            detect_questions = request.form.get('detect_questions', 'false').lower() == 'true'
            
            try:
                if file_path.lower().endswith('.pdf'):
                    result = ocr_engine.pdf_to_text(file_path, detect_questions)
                else:
                    result = ocr_engine.process_image(file_path, detect_questions)
                
                # Store results in session for further processing
                session['last_ocr_result'] = result
                session['last_file_path'] = file_path
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        return jsonify({'error': 'Invalid file type'}), 400
        
    return render_template('ocr.html')

@app.route('/map-answers', methods=['GET', 'POST'])
def map_answers():
    if request.method == 'POST':
        if 'last_ocr_result' not in session:
            return jsonify({'error': 'No OCR result available. Please scan a document first.'}), 400
            
        ocr_result = session.get('last_ocr_result')
        
        # Get question paper if provided
        question_paper = None
        if 'question_paper_id' in request.form:
            paper_id = request.form['question_paper_id']
            # In a real app, retrieve question paper from database
            # For now, just use a session variable if it exists
            if 'generated_paper' in session:
                question_paper = {q['number']: q['text'] for section in session['generated_paper']['sections'] 
                                 for q in section['questions']}
        
        # Map answers
        full_text = ocr_result.get('full_text', '')
        mapped_answers = answer_mapper.map_answers_to_questions(full_text, question_paper)
        
        # Store for grading
        session['mapped_answers'] = mapped_answers
        
        return jsonify({'mapped_answers': mapped_answers})
        
    return render_template('mapping.html')

@app.route('/grade', methods=['GET', 'POST'])
def grade_answers():
    if request.method == 'POST':
        if 'mapped_answers' not in session:
            return jsonify({'error': 'No mapped answers available. Please map answers first.'}), 400
        
        mapped_answers = session.get('mapped_answers')
        grading_results = {}
        
        # Get reference answers if available
        reference_answers = {}
        if 'generated_paper' in session:
            reference_answers = {q['number']: q['answer'] for section in session['generated_paper']['sections'] 
                               for q in section['questions']}
        
        # Get manually provided reference answers
        manual_references = request.json.get('reference_answers', {})
        reference_answers.update(manual_references)
        
        # Grade each answer
        for q_num, qa_pair in mapped_answers.items():
            student_answer = qa_pair.get('answer', '')
            question_text = qa_pair.get('question', '')
            
            # Find reference answer
            reference_answer = reference_answers.get(int(q_num) if isinstance(q_num, str) else q_num, '')
            if not reference_answer:
                reference_answer = request.json.get(f'reference_{q_num}', '')
            
            # Identify cognitive level
            cognitive_level = grading_system.identify_cognitive_level(question_text)
            
            # Grade
            if student_answer and reference_answer:
                result = grading_system.grade_answer(
                    student_answer, 
                    reference_answer, 
                    cognitive_level,
                    max_score=int(request.json.get(f'max_score_{q_num}', 10))
                )
                grading_results[q_num] = result
            else:
                grading_results[q_num] = {
                    'score': 0,
                    'max_score': int(request.json.get(f'max_score_{q_num}', 10)),
                    'percentage': 0,
                    'cognitive_level': cognitive_level,
                    'feedback': 'Cannot grade without both student answer and reference answer.'
                }
        
        # Store grading results
        session['grading_results'] = grading_results
        
        return jsonify({'grading_results': grading_results})
        
    return render_template('grading.html')

@app.route('/generate-question', methods=['GET', 'POST'])
def generate_question_paper():
    if request.method == 'POST':
        try:
            data = request.json
            
            # Create blueprint from form data
            blueprint = {
                'total_marks': int(data.get('total_marks', 100)),
                'time_hours': float(data.get('time_hours', 3)),
                'sections': []
            }
            
            # Process sections
            for section_data in data.get('sections', []):
                section = {
                    'name': section_data.get('name', 'Section'),
                    'description': section_data.get('description', ''),
                    'cognitive_levels': {},
                    'question_count': int(section_data.get('question_count', 5)),
                    'marks_per_question': int(section_data.get('marks_per_question', 10))
                }
                
                # Process cognitive levels
                for level, value in section_data.get('cognitive_levels', {}).items():
                    if value:
                        section['cognitive_levels'][level] = float(value)
                
                # Normalize cognitive levels to sum to 1.0
                if section['cognitive_levels']:
                    total = sum(section['cognitive_levels'].values())
                    if total > 0:
                        for level in section['cognitive_levels']:
                            section['cognitive_levels'][level] /= total
                
                blueprint['sections'].append(section)
            
            # Generate paper
            paper = question_generator.generate_question_paper(blueprint, data.get('subject'))
            
            # Store for later use
            session['generated_paper'] = paper
            
            return jsonify({'question_paper': paper})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('question_generator.html')

@app.route('/analytics', methods=['GET'])
def analytics():
    if 'grading_results' not in session:
        return render_template('analytics.html', has_data=False)
        
    grading_results = session.get('grading_results', {})
    
    # Prepare data for visualization
    scores = []
    questions = []
    percentages = []
    cognitive_levels = []
    
    for q_num, result in grading_results.items():
        scores.append(result.get('score', 0))
        questions.append(f'Q{q_num}')
        percentages.append(result.get('percentage', 0))
        cognitive_levels.append(result.get('cognitive_level', 'unknown'))
    
    # Create score distribution chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(questions, scores)
    ax.set_xlabel('Questions')
    ax.set_ylabel('Score')
    ax.set_title('Score Distribution by Question')
    
    # Save to base64 for embedding in HTML
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    score_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # Create cognitive level performance chart
    level_data = {}
    for level, score in zip(cognitive_levels, percentages):
        if level not in level_data:
            level_data[level] = []
        level_data[level].append(score)
    
    level_avg = {level: sum(scores)/len(scores) for level, scores in level_data.items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(level_avg.keys(), level_avg.values())
    ax.set_xlabel('Cognitive Level')
    ax.set_ylabel('Average Score (%)')
    ax.set_title('Performance by Cognitive Level')
    
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    level_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # Calculate overall statistics
    total_score = sum(scores)
    max_possible = sum([result.get('max_score', 10) for result in grading_results.values()])
    average_percentage = (total_score / max_possible * 100) if max_possible else 0
    
    stats = {
        'total_score': total_score,
        'max_possible': max_possible,
        'average_percentage': average_percentage,
        'question_count': len(questions)
    }
    
    return render_template(
        'analytics.html',
        has_data=True,
        score_chart=score_chart,
        level_chart=level_chart,
        stats=stats,
        results=grading_results
    )

@app.route('/export-results', methods=['POST'])
def export_results():
    if 'grading_results' not in session:
        return jsonify({'error': 'No grading results available'}), 400
        
    format_type = request.form.get('format', 'json')
    grading_results = session.get('grading_results', {})
    
    if format_type == 'json':
        response = jsonify(grading_results)
        response.headers['Content-Disposition'] = 'attachment; filename=grading_results.json'
        return response
        
    elif format_type == 'csv':
        # Convert to DataFrame
        rows = []
        for q_num, result in grading_results.items():
            rows.append({
                'Question': f'Q{q_num}',
                'Score': result.get('score', 0),
                'Maximum': result.get('max_score', 0),
                'Percentage': result.get('percentage', 0),
                'Cognitive Level': result.get('cognitive_level', ''),
                'Feedback': result.get('feedback', '')
            })
            
        df = pd.DataFrame(rows)
        
        # Convert to CSV
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=grading_results.csv'
        }
        
    return jsonify({'error': 'Unsupported format'}), 400

@app.route('/batch-process', methods=['POST'])
def batch_process():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
        
    # Get reference question paper
    question_paper = None
    if 'generated_paper' in session:
        question_paper = {q['number']: q['text'] for section in session['generated_paper']['sections'] 
                         for q in section['questions']}
        
        reference_answers = {q['number']: q['answer'] for section in session['generated_paper']['sections'] 
                           for q in section['questions']}
    else:
        return jsonify({'error': 'No question paper available. Please generate one first.'}), 400
    
    # Process each file
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + filename)
            file.save(file_path)
            
            # Process with OCR
            try:
                if file_path.lower().endswith('.pdf'):
                    ocr_result = ocr_engine.pdf_to_text(file_path, True)
                else:
                    ocr_result = ocr_engine.process_image(file_path, True)
                
                # Map answers
                full_text = ocr_result.get('full_text', '')
                mapped_answers = answer_mapper.map_answers_to_questions(full_text, question_paper)
                
                # Grade answers
                grading_results = {}
                for q_num, qa_pair in mapped_answers.items():
                    student_answer = qa_pair.get('answer', '')
                    question_text = qa_pair.get('question', '')
                    
                    # Find reference answer
                    ref_q_num = int(q_num) if isinstance(q_num, str) else q_num
                    reference_answer = reference_answers.get(ref_q_num, '')
                    
                    # Identify cognitive level
                    cognitive_level = grading_system.identify_cognitive_level(question_text)
                    
                    # Grade
                    if student_answer and reference_answer:
                        result = grading_system.grade_answer(
                            student_answer, 
                            reference_answer, 
                            cognitive_level
                        )
                        grading_results[q_num] = result
                    else:
                        grading_results[q_num] = {
                            'score': 0,
                            'max_score': 10,
                            'percentage': 0,
                            'cognitive_level': cognitive_level,
                            'feedback': 'Cannot grade without both student answer and reference answer.'
                        }
                
                # Calculate overall score
                total_score = sum([r.get('score', 0) for r in grading_results.values()])
                max_possible = sum([r.get('max_score', 0) for r in grading_results.values()])
                percentage = (total_score / max_possible * 100) if max_possible else 0
                
                # Add to results
                results.append({
                    'filename': filename,
                    'total_score': total_score,
                    'max_score': max_possible,
                    'percentage': percentage,
                    'details': grading_results
                })
                
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
        else:
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type'
            })
    
    return jsonify({'batch_results': results})

if __name__ == '__main__':
    app.run(debug=True)
