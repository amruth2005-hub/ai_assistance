# grading.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
        
        # Grading weights based on cognitive levels
        self.cognitive_weights = {
            'remember': {'exact_match': 0.9, 'concept_match': 0.1},
            'understand': {'exact_match': 0.7, 'concept_match': 0.3},
            'apply': {'exact_match': 0.6, 'concept_match': 0.4},
            'analyze': {'exact_match': 0.5, 'concept_match': 0.5},
            'evaluate': {'exact_match': 0.3, 'concept_match': 0.7},
            'create': {'exact_match': 0.2, 'concept_match': 0.8}
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def identify_cognitive_level(self, question_text):
        """Identify the cognitive level of a question based on Bloom's Taxonomy"""
        if not question_text:
            return 'understand'
        
        question = self.preprocess_text(question_text)
        question_words = question.split()
        
        # Count occurrences of level-specific keywords
        level_scores = {}
        
        for level, keywords in self.bloom_keywords.items():
            score = 0
            for keyword in keywords:
                # Check for exact word matches
                if keyword in question_words:
                    score += 2
                # Check for partial matches
                elif any(keyword in word for word in question_words):
                    score += 1
            level_scores[level] = score
        
        # Return highest scoring level, defaulting to 'understand' if no match
        if not level_scores or max(level_scores.values()) == 0:
            return 'understand'
        
        max_level = max(level_scores.items(), key=lambda x: x[1])
        return max_level[0]
    
    def calculate_similarity(self, student_answer, reference_answer):
        """Calculate semantic similarity between student answer and reference"""
        if not student_answer or not reference_answer:
            return 0.0
        
        # Prepare the texts
        student_clean = self.preprocess_text(student_answer)
        reference_clean = self.preprocess_text(reference_answer)
        
        if not student_clean or not reference_clean:
            return 0.0
        
        # Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform([student_clean, reference_clean])
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(cosine_sim)
        except:
            # Fallback for very short texts
            return self.calculate_keyword_overlap(student_clean, reference_clean)
    
    def calculate_keyword_overlap(self, student_text, reference_text):
        """Calculate keyword overlap ratio as fallback similarity measure"""
        student_words = set(w for w in student_text.split() if w not in self.stopwords and len(w) > 2)
        ref_words = set(w for w in reference_text.split() if w not in self.stopwords and len(w) > 2)
        
        if not ref_words:
            return 0.0
            
        common_words = student_words.intersection(ref_words)
        return len(common_words) / len(ref_words)
    
    def calculate_concept_score(self, student_answer, cognitive_level):
        """Calculate score based on cognitive level specific concepts"""
        if cognitive_level not in self.bloom_keywords:
            return 0.0
        
        student_text = self.preprocess_text(student_answer)
        relevant_keywords = self.bloom_keywords[cognitive_level]
        
        # Check for presence of level-specific keywords
        keyword_matches = sum(1 for kw in relevant_keywords if kw in student_text.split())
        
        # Normalize by expected keywords for this level
        max_expected = min(len(relevant_keywords) // 3, 3)  # Expect at most 1/3 of keywords
        if max_expected == 0:
            max_expected = 1
        
        return min(keyword_matches / max_expected, 1.0)
    
    def grade_answer(self, student_answer, reference_answer, cognitive_level=None, max_score=10, config=None):
        """Grade an answer based on similarity, cognitive level, and max score"""
        if not student_answer:
            return {
                'score': 0,
                'max_score': max_score,
                'percentage': 0,
                'similarity': 0,
                'cognitive_level': cognitive_level or 'understand',
                'feedback': 'No answer provided.'
            }
        
        if not reference_answer:
            return {
                'score': 0,
                'max_score': max_score,
                'percentage': 0,
                'similarity': 0,
                'cognitive_level': cognitive_level or 'understand',
                'feedback': 'Cannot grade without reference answer.'
            }
        
        # Identify cognitive level if not provided
        if not cognitive_level:
            cognitive_level = 'understand'
        
        # Get configuration settings
        if not config:
            config = {
                'strictness': 3,
                'keyword_matching': True,
                'partial_credit': True,
                'detailed_feedback': True
            }
        
        # Calculate base similarity score
        base_similarity = self.calculate_similarity(student_answer, reference_answer)
        
        # Calculate concept score for higher cognitive levels
        concept_score = self.calculate_concept_score(student_answer, cognitive_level)
        
        # Get weights for this cognitive level
        weights = self.cognitive_weights.get(cognitive_level, {'exact_match': 0.7, 'concept_match': 0.3})
        
        # Calculate weighted score
        weighted_score = (base_similarity * weights['exact_match'] + 
                         concept_score * weights['concept_match'])
        
        # Apply strictness adjustment
        strictness = float(config.get('strictness', 3))
        if strictness < 3:
            # Lenient grading - boost scores
            weighted_score = weighted_score ** 0.8
        elif strictness > 3:
            # Strict grading - reduce scores
            weighted_score = weighted_score ** 1.2
        
        # Apply partial credit settings
        if not config.get('partial_credit', True):
            # No partial credit - threshold at 70%
            if weighted_score < 0.7:
                weighted_score = 0
        
        # Calculate final score
        final_score = weighted_score * max_score
        
        # Round to nearest 0.5
        rounded_score = round(final_score * 2) / 2
        
        # Ensure score doesn't exceed max
        rounded_score = min(rounded_score, max_score)
        
        # Calculate percentage
        percentage = (rounded_score / max_score) * 100 if max_score > 0 else 0
        
        # Generate feedback
        feedback = self.generate_feedback(
            student_answer, 
            reference_answer, 
            rounded_score, 
            max_score, 
            cognitive_level,
            config.get('detailed_feedback', True)
        )
        
        return {
            'score': rounded_score,
            'max_score': max_score,
            'percentage': percentage,
            'similarity': base_similarity,
            'concept_score': concept_score,
            'cognitive_level': cognitive_level,
            'feedback': feedback
        }
    
    def generate_feedback(self, student_answer, reference_answer, score, max_score, cognitive_level, detailed=True):
        """Generate constructive feedback based on the grading results"""
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        # Basic feedback based on score percentage
        if percentage >= 90:
            base_feedback = "Excellent work! Your answer demonstrates strong mastery of the content."
        elif percentage >= 80:
            base_feedback = "Very good answer. You've shown good understanding of the key concepts."
        elif percentage >= 70:
            base_feedback = "Good answer with most key points covered."
        elif percentage >= 60:
            base_feedback = "Satisfactory answer. Some important concepts were included, but others were missing."
        elif percentage >= 50:
            base_feedback = "Your answer addresses some aspects of the question, but needs improvement."
        elif percentage > 0:
            base_feedback = "Your answer shows some effort, but significant improvement is needed."
        else:
            base_feedback = "No answer was provided or the answer was not relevant to the question."
        
        if not detailed:
            return base_feedback
        
        # Add cognitive level specific feedback
        level_feedback = ""
        if cognitive_level == 'remember' and percentage < 80:
            level_feedback = " Focus on including more specific terms, definitions, and factual information."
        elif cognitive_level == 'understand' and percentage < 80:
            level_feedback = " Work on explaining concepts more clearly and showing connections between ideas."
        elif cognitive_level == 'apply' and percentage < 80:
            level_feedback = " Demonstrate how to apply these concepts to solve specific problems or scenarios."
        elif cognitive_level == 'analyze' and percentage < 80:
            level_feedback = " Break down the concepts into components and analyze relationships more thoroughly."
        elif cognitive_level == 'evaluate' and percentage < 80:
            level_feedback = " Strengthen your critical evaluation with more supporting evidence and reasoning."
        elif cognitive_level == 'create' and percentage < 80:
            level_feedback = " Develop more original ideas and show innovative integration of concepts."
        
        # Add specific content suggestions if score is below 70%
        content_feedback = ""
        if percentage < 70:
            # Extract key terms from reference that might be missing
            ref_clean = self.preprocess_text(reference_answer)
            student_clean = self.preprocess_text(student_answer)
            
            ref_words = set(w for w in ref_clean.split() if w not in self.stopwords and len(w) > 3)
            student_words = set(w for w in student_clean.split() if w not in self.stopwords and len(w) > 3)
            
            missing_concepts = list(ref_words - student_words)[:3]  # Limit to 3 concepts
            
            if missing_concepts:
                content_feedback = f" Consider addressing these key concepts: {', '.join(missing_concepts)}."
        
        return base_feedback + level_feedback + content_feedback
    
    def batch_grade(self, student_answers, reference_answers, cognitive_levels=None, max_scores=None, config=None):
        """Grade multiple answers at once"""
        if not isinstance(student_answers, dict) or not isinstance(reference_answers, dict):
            raise ValueError("Answers must be provided as dictionaries")
        
        results = {}
        
        for question_id in student_answers:
            student_answer = student_answers.get(question_id, "")
            reference_answer = reference_answers.get(question_id, "")
            cognitive_level = cognitive_levels.get(question_id) if cognitive_levels else None
            max_score = max_scores.get(question_id, 10) if max_scores else 10
            
            result = self.grade_answer(
                student_answer,
                reference_answer,
                cognitive_level,
                max_score,
                config
            )
            
            results[question_id] = result
        
        return results
    
    def calculate_overall_statistics(self, grading_results):
        """Calculate overall statistics from grading results"""
        if not grading_results:
            return {}
        
        scores = [result['score'] for result in grading_results.values()]
        max_scores = [result['max_score'] for result in grading_results.values()]
        percentages = [result['percentage'] for result in grading_results.values()]
        
        total_score = sum(scores)
        total_max = sum(max_scores)
        
        statistics = {
            'total_score': total_score,
            'total_max': total_max,
            'overall_percentage': (total_score / total_max * 100) if total_max > 0 else 0,
            'average_percentage': sum(percentages) / len(percentages) if percentages else 0,
            'highest_score': max(percentages) if percentages else 0,
            'lowest_score': min(percentages) if percentages else 0,
            'question_count': len(grading_results),
            'cognitive_breakdown': self._calculate_cognitive_breakdown(grading_results)
        }
        
        return statistics
    
    def _calculate_cognitive_breakdown(self, grading_results):
        """Calculate performance breakdown by cognitive level"""
        breakdown = {}
        
        for result in grading_results.values():
            level = result.get('cognitive_level', 'unknown')
            if level not in breakdown:
                breakdown[level] = {'scores': [], 'count': 0}
            
            breakdown[level]['scores'].append(result['percentage'])
            breakdown[level]['count'] += 1
        
        # Calculate averages
        for level in breakdown:
            scores = breakdown[level]['scores']
            breakdown[level]['average'] = sum(scores) / len(scores) if scores else 0
        
        return breakdown


# Example usage and testing
if __name__ == "__main__":
    # Initialize the grading system
    grader = GradingSystem()
    
    # Example test case
    student_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
    reference_answer = "Machine learning is a branch of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed."
    
    # Grade the answer
    result = grader.grade_answer(
        student_answer=student_answer,
        reference_answer=reference_answer,
        cognitive_level='understand',
        max_score=10
    )
    
    print("Grading Result:")
    print(f"Score: {result['score']}/{result['max_score']}")
    print(f"Percentage: {result['percentage']:.1f}%")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Feedback: {result['feedback']}")
