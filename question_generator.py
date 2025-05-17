# question_generator.py
import random
import json
import os
from datetime import datetime

class QuestionPaperGenerator:
    """Generates structured question papers based on a blueprint and question bank"""
    
    def __init__(self, question_bank_path=None):
        self.question_bank = self.load_question_bank(question_bank_path)
        
        # Default blueprint structure
        self.default_blueprint = {
            'total_marks': 100,
            'time_hours': 3,
            'sections': [
                {
                    'name': 'Section A',
                    'description': 'Answer all questions',
                    'cognitive_levels': {
                        'remember': 0.4,
                        'understand': 0.4,
                        'apply': 0.2
                    },
                    'question_count': 10,
                    'marks_per_question': 2
                },
                {
                    'name': 'Section B',
                    'description': 'Answer any 5 out of 7 questions',
                    'cognitive_levels': {
                        'understand': 0.3,
                        'apply': 0.4,
                        'analyze': 0.3
                    },
                    'question_count': 7,
                    'marks_per_question': 6
                },
                {
                    'name': 'Section C',
                    'description': 'Answer any 2 out of 3 questions',
                    'cognitive_levels': {
                        'analyze': 0.3,
                        'evaluate': 0.4,
                        'create': 0.3
                    },
                    'question_count': 3,
                    'marks_per_question': 15
                }
            ]
        }
    
    def load_question_bank(self, file_path=None):
        """Load or create a question bank"""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except:
                print(f"Error loading question bank from {file_path}")
        
        # Return a sample question bank if file doesn't exist
        return {
            'subject': 'Sample Subject',
            'questions': {
                'remember': [
                    {'id': 'r1', 'text': 'Define artificial intelligence.', 'answer': 'Artificial Intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.'},
                    {'id': 'r2', 'text': 'List the main components of an OCR system.', 'answer': 'The main components of an OCR system include image acquisition, preprocessing, feature extraction, classification, and post-processing.'}
                ],
                'understand': [
                    {'id': 'u1', 'text': 'Explain how machine learning differs from traditional programming.', 'answer': 'In traditional programming, developers write explicit rules for the computer to follow, whereas in machine learning, algorithms learn patterns from data to make decisions without being explicitly programmed.'},
                    {'id': 'u2', 'text': 'Compare supervised and unsupervised learning approaches.', 'answer': 'Supervised learning uses labeled data where the algorithm learns to predict outputs based on input-output pairs, while unsupervised learning works with unlabeled data to find patterns or groupings without predefined correct answers.'}
                ],
                'apply': [
                    {'id': 'a1', 'text': 'Demonstrate how to normalize a dataset with examples.', 'answer': 'To normalize a dataset, you calculate (x - min)/(max - min) for each value. For example, with values [10, 20, 30], min=10, max=30, normalized values would be [0, 0.5, 1].'},
                    {'id': 'a2', 'text': 'Solve the following classification problem using the decision tree algorithm.', 'answer': 'To solve this using a decision tree: 1) Choose the attribute with highest information gain for the root node, 2) Create branches for each value, 3) Repeat process recursively for each branch until all data is classified or no attributes remain.'}
                ],
                'analyze': [
                    {'id': 'an1', 'text': 'Analyze the strengths and weaknesses of using neural networks for image recognition.', 'answer': 'Strengths: High accuracy, ability to learn complex patterns, automatic feature extraction. Weaknesses: Requires large amounts of data, computationally expensive, black-box nature makes interpretability difficult, and susceptibility to adversarial examples.'},
                    {'id': 'an2', 'text': 'Differentiate between precision and recall in the context of model evaluation.', 'answer': 'Precision measures how many of the positively classified items are actually positive (true positives/all predicted positives). Recall measures how many of the actual positive items were correctly identified (true positives/all actual positives). Precision focuses on false positives, while recall focuses on false negatives.'}
                ],
                'evaluate': [
                    {'id': 'e1', 'text': 'Evaluate the ethical implications of using facial recognition in public spaces.', 'answer': 'Ethical implications include: privacy concerns as individuals are tracked without consent, potential for discrimination if the system has biases against certain demographics, security benefits for identifying threats, civil liberty concerns regarding surveillance, and questions about data ownership and usage.'},
                    {'id': 'e2', 'text': 'Justify why ensemble methods often perform better than individual models.', 'answer': 'Ensemble methods perform better because they: reduce variance by averaging multiple models, reduce bias by combining different model types, are less likely to overfit as errors cancel out, capture more complex relationships in the data, and are more robust to noise and outliers in the training data.'}
                ],
                'create': [
                    {'id': 'c1', 'text': 'Design a system architecture for an AI-powered educational assessment tool.', 'answer': 'An AI-powered educational assessment architecture would include: a user interface layer, authentication and user management, content repository for questions and resources, AI processing layer with NLP for text analysis and computer vision for diagrams, assessment engine to evaluate responses, personalization module to adapt to student needs, analytics dashboard for instructors, and secure API interfaces.'},
                    {'id': 'c2', 'text': 'Propose a novel approach to improve the accuracy of sentiment analysis on social media posts.', 'answer': 'A novel approach could include: multimodal analysis combining text, emojis, and images; contextual understanding through conversation flows; cultural and demographic adaptation layers; sarcasm and irony detection using contrast patterns; emotion intensity scoring beyond binary sentiment; and federated learning to preserve privacy while improving with user feedback.'}
                ]
            }
        }
    
    def add_question_to_bank(self, level, question_text, answer_text):
        """Add a new question to the question bank"""
        if level not in self.question_bank['questions']:
            self.question_bank['questions'][level] = []
            
        new_id = f"{level[0]}{len(self.question_bank['questions'][level]) + 1}"
        
        new_question = {
            'id': new_id,
            'text': question_text,
            'answer': answer_text
        }
        
        self.question_bank['questions'][level].append(new_question)
        return new_id
    
    def save_question_bank(self, file_path):
        """Save the current question bank to a file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.question_bank, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving question bank: {e}")
            return False
    
    def generate_question_paper(self, blueprint=None, subject=None):
        """Generate a question paper based on the provided blueprint"""
        if blueprint is None:
            blueprint = self.default_blueprint
            
        if subject is None and 'subject' in self.question_bank:
            subject = self.question_bank['subject']
        else:
            subject = "Examination"
            
        question_paper = {
            'title': f"{subject} Question Paper",
            'date': datetime.now().strftime("%B %d, %Y"),
            'total_marks': blueprint['total_marks'],
            'time_hours': blueprint['time_hours'],
            'sections': []
        }
        
        for section_blueprint in blueprint['sections']:
            section = {
                'name': section_blueprint['name'],
                'description': section_blueprint['description'],
                'questions': []
            }
            
            # Calculate how many questions to select from each cognitive level
            level_counts = {}
            for level, proportion in section_blueprint['cognitive_levels'].items():
                level_counts[level] = round(section_blueprint['question_count'] * proportion)
                
            # Adjust to ensure we get exactly the right number of questions
            total_selected = sum(level_counts.values())
            
            if total_selected < section_blueprint['question_count']:
                # Add the remaining questions to the highest weighted level
                max_level = max(section_blueprint['cognitive_levels'].items(), key=lambda x: x[1])[0]
                level_counts[max_level] += section_blueprint['question_count'] - total_selected
            elif total_selected > section_blueprint['question_count']:
                # Remove excess questions from the lowest weighted level
                min_level = min(section_blueprint['cognitive_levels'].items(), key=lambda x: x[1])[0]
                level_counts[min_level] -= total_selected - section_blueprint['question_count']
            
            # Select questions for each level
            question_number = 1
            for level, count in level_counts.items():
                if count <= 0:
                    continue
                    
                # Check if we have enough questions in the bank
                available = self.question_bank['questions'].get(level, [])
                if len(available) < count:
                    print(f"Warning: Not enough {level} questions in the bank. Needed {count}, have {len(available)}.")
                    # Fill with placeholder questions if needed
                    for i in range(len(available), count):
                        self.add_question_to_bank(
                            level, 
                            f"[PLACEHOLDER] {level.capitalize()} level question {i+1}", 
                            f"This is a placeholder answer for a {level} level question."
                        )
                
                # Select random questions without replacement
                selected = random.sample(self.question_bank['questions'].get(level, []), count)
                
                for q in selected:
                    section['questions'].append({
                        'number': question_number,
                        'text': q['text'],
                        'marks': section_blueprint['marks_per_question'],
                        'cognitive_level': level,
                        'answer': q['answer']  # Include model answer for grading
                    })
                    question_number += 1
            
            question_paper['sections'].append(section)
        
        return question_paper
    
    def export_question_paper(self, question_paper, output_format='json', file_path=None):
        """Export the question paper in various formats"""
        if output_format == 'json':
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(question_paper, f, indent=2)
                return file_path
            else:
                return json.dumps(question_paper, indent=2)
                
        elif output_format == 'text':
            output = []
            output.append(f"{question_paper['title']}")
            output.append(f"Date: {question_paper['date']}")
            output.append(f"Total Marks: {question_paper['total_marks']} | Time: {question_paper['time_hours']} Hours\n")
            
            for section in question_paper['sections']:
                output.append(f"{section['name']}")
                output.append(f"{section['description']}\n")
                
                for q in section['questions']:
                    output.append(f"{q['number']}. {q['text']} ({q['marks']} marks)")
                
                output.append("")  # Empty line between sections
            
            text_output = "\n".join(output)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(text_output)
                return file_path
            else:
                return text_output
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
