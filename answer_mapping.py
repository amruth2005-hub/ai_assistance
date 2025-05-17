# answer_mapping.py
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)

class AnswerMapper:
    """Maps extracted text to corresponding questions based on identifiers and structure"""
    
    def __init__(self):
        self.question_patterns = [
            r'(?:^|\n)(\d+[\.)]\s*.*?)(?=\n\d+[\.)]\s*|\Z)', # Standard numbered questions 
            r'(?:^|\n)(Q\.?\s*\d+\.?\s*.*?)(?=\n\s*Q\.?\s*\d+\.?\s*|\Z)', # Q1, Q.1 format
            r'(?:^|\n)(Question\s*\d+\.?\s*.*?)(?=\n\s*Question\s*\d+\.?\s*|\Z)' # Question 1 format
        ]
    
    def extract_questions_and_answers(self, text):
        """Extract questions and answers from OCR text"""
        results = {}
        
        # Try each pattern until we find one that works
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            matched_items = list(matches)
            
            if matched_items:
                for idx, match in enumerate(matched_items):
                    # Get the question text
                    full_text = match.group(1).strip()
                    
                    # Try to separate question from answer
                    question_parts = re.split(r'\n{2,}', full_text, maxsplit=1)
                    
                    if len(question_parts) > 1:
                        question = question_parts[0].strip()
                        answer = question_parts[1].strip()
                    else:
                        # If separation is unclear, try another approach
                        sentences = sent_tokenize(full_text)
                        if len(sentences) > 1:
                            question = sentences[0].strip()
                            answer = ' '.join(sentences[1:]).strip()
                        else:
                            question = full_text
                            answer = ""
                    
                    # Extract question number
                    q_num_match = re.match(r'(?:Q\.?|Question\s*)?(\d+)[\.\)]*\s*', question)
                    if q_num_match:
                        q_num = int(q_num_match.group(1))
                    else:
                        q_num = idx + 1
                    
                    results[q_num] = {
                        'question': question,
                        'answer': answer
                    }
                
                # If we found matches, stop trying patterns
                if results:
                    break
        
        return results
    
    def map_answers_to_questions(self, ocr_text, question_paper=None):
        """Map extracted answers to known questions"""
        # Extract what we can from OCR text
        extracted_qa = self.extract_questions_and_answers(ocr_text)
        
        # If we have a reference question paper, use it to improve mapping
        if question_paper:
            mapped_answers = {}
            for q_num, q_text in question_paper.items():
                if q_num in extracted_qa:
                    # Direct match by question number
                    mapped_answers[q_num] = {
                        'question': q_text,
                        'answer': extracted_qa[q_num]['answer']
                    }
                else:
                    # Try to find by similarity
                    # This would use more advanced text similarity in a full implementation
                    mapped_answers[q_num] = {
                        'question': q_text,
                        'answer': "Not found"
                    }
            
            return mapped_answers
        
        return extracted_qa

# grading.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('stopwords', quiet=True)

class GradingSystem:
    """Evaluates answers based on reference material and grading criteria"""
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Bloom's Taxonomy keywords for different cognitive levels
        self.bloom_keywords = {
            'remember': ['define', 'list', 'recall', 'name', 'identify', 'state', 'select', 'match', 'recognize'],
            'understand': ['explain', 'interpret', 'describe', 'compare', 'discuss', 'distinguish', 'predict', 'associate'],
            'apply': ['apply', 'demonstrate', 'calculate', 'complete', 'illustrate', 'show', 'solve', 'use', 'examine'],
            'analyze': ['analyze', 'differentiate', 'organize', 'relate', 'compare', 'contrast', 'distinguish', 'examine'],
            'evaluate': ['evaluate', 'assess', 'justify', 'critique', 'judge', 'defend', 'criticize'],
            'create': ['create', 'design', 'formulate', 'construct', 'propose', 'develop', 'invent', 'compose']
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def identify_cognitive_level(self, question_text):
        """Identify the cognitive level of a question based on Bloom's Taxonomy"""
        question = self.preprocess_text(question_text)
        
        # Count occurrences of level-specific keywords
        level_scores = {}
        
        for level, keywords in self.bloom_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question.split())
            level_scores[level] = score
        
        # Return highest scoring level, defaulting to 'remember' if no match
        max_level = max(level_scores.items(), key=lambda x: x[1])
        if max_level[1] > 0:
            return max_level[0]
        return 'remember'
    
    def calculate_similarity(self, student_answer, reference_answer):
        """Calculate semantic similarity between student answer and reference"""
        if not student_answer or not reference_answer:
            return 0.0
        
        # Prepare the texts
        student_clean = self.preprocess_text(student_answer)
        reference_clean = self.preprocess_text(reference_answer)
        
        # Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform([student_clean, reference_clean])
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return cosine_sim
        except:
            # Fallback for very short texts
            return self.calculate_keyword_overlap(student_clean, reference_clean)
    
    def calculate_keyword_overlap(self, student_text, reference_text):
        """Calculate keyword overlap ratio as fallback similarity measure"""
        student_words = set(w for w in student_text.split() if w not in self.stopwords)
        ref_words = set(w for w in reference_text.split() if w not in self.stopwords)
        
        if not ref_words:
            return 0.0
            
        common_words = student_words.intersection(ref_words)
        return len(common_words) / len(ref_words)
    
    def grade_answer(self, student_answer, reference_answer, cognitive_level, max_score=10):
        """Grade an answer based on similarity, cognitive level, and max score"""
        # Calculate basic similarity score
        base_similarity = self.calculate_similarity(student_answer, reference_answer)
        
        # Adjust based on cognitive level (higher levels get more focus on concepts than exact matches)
        level_weights = {
            'remember': 1.0,      # Exact matching is important
            'understand': 0.9,    # Some flexibility in wording
            'apply': 0.8,         # Application focus, less on wording
            'analyze': 0.7,       # Concepts matter more than wording
            'evaluate': 0.6,      # Reasoning matters most
            'create': 0.5         # Creativity and originality matter most
        }
        
        # Get weight for exact matching based on cognitive level
        exact_weight = level_weights.get(cognitive_level, 0.8)
        
        # For higher cognitive levels, we also check for presence of level-specific keywords
        level_keyword_score = 0
        if cognitive_level in self.bloom_keywords:
            relevant_keywords = self.bloom_keywords[cognitive_level]
            student_words = set(self.preprocess_text(student_answer).split())
            keyword_matches = sum(1 for kw in relevant_keywords if kw in student_words)
            level_keyword_score = min(keyword_matches / max(len(relevant_keywords)/3, 1), 1.0)
        
        # Final score calculation
        concept_weight = 1.0 - exact_weight
        final_score = (base_similarity * exact_weight + level_keyword_score * concept_weight) * max_score
        
        # Round to nearest 0.5
        rounded_score = round(final_score * 2) / 2
        
        return {
            'score': rounded_score,
            'max_score': max_score,
            'percentage': (rounded_score / max_score) * 100,
            'similarity': base_similarity,
            'cognitive_level': cognitive_level,
            'feedback': self.generate_feedback(student_answer, reference_answer, rounded_score, max_score, cognitive_level)
        }
    
    def generate_feedback(self, student_answer, reference_answer, score, max_score, cognitive_level):
        """Generate constructive feedback based on the grading results"""
        percentage = (score / max_score) * 100
        
        # Basic feedback template based on score percentage
        if percentage >= 90:
            feedback = "Excellent work! Your answer demonstrates strong mastery of the content."
        elif percentage >= 80:
            feedback = "Very good answer. You've shown good understanding of the key concepts."
        elif percentage >= 70:
            feedback = "Good answer with most key points covered."
        elif percentage >= 60:
            feedback = "Satisfactory answer. Some important concepts were included, but others were missing."
        elif percentage >= 50:
            feedback = "Your answer addresses some aspects of the question, but needs improvement."
        else:
            feedback = "Your answer needs significant improvement. Please review the course material."
        
        # Add suggestion for improvement based on cognitive level and missing content
        if cognitive_level == 'remember' and percentage < 80:
            feedback += " Try to include more key terms and definitions in your answer."
        elif cognitive_level == 'understand' and percentage < 80:
            feedback += " Try to explain the concepts in more detail and show connections between ideas."
        elif cognitive_level == 'apply' and percentage < 80:
            feedback += " Focus on demonstrating how to apply these concepts to solve problems."
        elif cognitive_level == 'analyze' and percentage < 80:
            feedback += " Work on breaking down the concepts into their components and analyzing relationships."
        elif cognitive_level == 'evaluate' and percentage < 80:
            feedback += " Develop your critical evaluation with more supporting evidence."
        elif cognitive_level == 'create' and percentage < 80:
            feedback += " Try to develop more original ideas and integrate concepts in novel ways."
        
        # Add specific content suggestions if score is below 70%
        if percentage < 70:
            # Extract important phrases from reference answer that are missing from student answer
            ref_clean = self.preprocess_text(reference_answer)
            student_clean = self.preprocess_text(student_answer)
            
            ref_unique_words = set(w for w in ref_clean.split() if w not in self.stopwords) - \
                             set(w for w in student_clean.split() if w not in self.stopwords)
            
            if ref_unique_words:
                missing_concepts = list(ref_unique_words)[:5]  # Limit to 5 concepts
                feedback += f" Consider including these key concepts: {', '.join(missing_concepts)}."
        
        return feedback
