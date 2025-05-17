# utils.py
import os
import re
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
import logging
from werkzeug.utils import secure_filename

# Set up logging
logger = logging.getLogger(__name__)

class FileHandler:
    """Utility class for file operations"""
    
    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        """Check if a file has an allowed extension"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    @staticmethod
    def secure_upload_filename(filename: str) -> str:
        """Generate a secure filename for uploads"""
        # Get file extension
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Generate unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Create secure filename
        base_name = secure_filename(filename.rsplit('.', 1)[0])
        secure_name = f"{timestamp}_{unique_id}_{base_name}"
        
        return f"{secure_name}.{ext}" if ext else secure_name
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            return ""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Ensure a directory exists, create if it doesn't"""
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False


class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_question_numbers(text: str) -> List[int]:
        """Extract question numbers from text"""
        # Common question number patterns
        patterns = [
            r'(?:^|\n)\s*(\d+)[\.\)]\s*',  # 1. or 1)
            r'(?:^|\n)\s*Q\.?\s*(\d+)[\.\)]?\s*',  # Q1. or Q.1 or Q1)
            r'(?:^|\n)\s*Question\s+(\d+)[\.\)]?\s*'  # Question 1
        ]
        
        numbers = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match.group(1))
                    numbers.add(num)
                except ValueError:
                    continue
        
        return sorted(list(numbers))
    
    @staticmethod
    def split_by_questions(text: str) -> Dict[int, str]:
        """Split text by question numbers"""
        # Pattern to match question headers
        question_pattern = r'(?:^|\n)\s*(?:Q\.?\s*|Question\s+)?(\d+)[\.\)]\s*'
        
        # Find all question matches
        matches = list(re.finditer(question_pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if not matches:
            return {}
        
        questions = {}
        for i, match in enumerate(matches):
            q_num = int(match.group(1))
            start_pos = match.end()
            
            # Find end position (start of next question or end of text)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # Extract question content
            content = text[start_pos:end_pos].strip()
            if content:
                questions[q_num] = content
        
        return questions
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        if not text:
            return 0
        return len(text.split())
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter words
        keywords = [
            word for word in words 
            if len(word) >= min_length and word not in stop_words
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords


class JSONHandler:
    """Utility class for JSON operations"""
    
    @staticmethod
    def save_json(data: Dict, file_path: str) -> bool:
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return None
    
    @staticmethod
    def merge_json_files(file_paths: List[str]) -> Dict:
        """Merge multiple JSON files into one dictionary"""
        merged = {}
        for file_path in file_paths:
            data = JSONHandler.load_json(file_path)
            if data:
                merged.update(data)
        return merged


class SessionManager:
    """Utility class for session management"""
    
    @staticmethod
    def store_result(session, key: str, data: Dict, expire_hours: int = 2) -> bool:
        """Store result in session with expiration"""
        try:
            timestamp = datetime.now().isoformat()
            session[key] = {
                'data': data,
                'timestamp': timestamp,
                'expire_hours': expire_hours
            }
            return True
        except Exception as e:
            logger.error(f"Error storing session data: {e}")
            return False
    
    @staticmethod
    def retrieve_result(session, key: str) -> Optional[Dict]:
        """Retrieve result from session if not expired"""
        try:
            if key not in session:
                return None
            
            stored = session[key]
            timestamp = datetime.fromisoformat(stored['timestamp'])
            expire_hours = stored.get('expire_hours', 2)
            
            # Check if expired
            if (datetime.now() - timestamp).total_seconds() > expire_hours * 3600:
                del session[key]
                return None
            
            return stored['data']
        except Exception as e:
            logger.error(f"Error retrieving session data: {e}")
            return None
    
    @staticmethod
    def clear_expired_sessions(session) -> None:
        """Clear expired session data"""
        keys_to_remove = []
        for key, value in session.items():
            if isinstance(value, dict) and 'timestamp' in value:
                try:
                    timestamp = datetime.fromisoformat(value['timestamp'])
                    expire_hours = value.get('expire_hours', 2)
                    
                    if (datetime.now() - timestamp).total_seconds() > expire_hours * 3600:
                        keys_to_remove.append(key)
                except Exception:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del session[key]


class ValidationHelper:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_score(score: Union[int, float], max_score: Union[int, float]) -> bool:
        """Validate if score is within valid range"""
        try:
            score = float(score)
            max_score = float(max_score)
            return 0 <= score <= max_score
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_cognitive_level(level: str) -> bool:
        """Validate Bloom's taxonomy cognitive level"""
        valid_levels = {
            'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
        }
        return level.lower() in valid_levels
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove potentially harmful characters
        text = re.sub(r'[<>"\']', '', text)
        
        # Limit length
        text = text[:max_length]
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class DateTimeHelper:
    """Utility class for date and time operations"""
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime object"""
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
        """Parse datetime string"""
        try:
            return datetime.strptime(dt_str, format_str)
        except ValueError as e:
            logger.error(f"Error parsing datetime {dt_str}: {e}")
            return None
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Get human readable time difference"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp as string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


class StatisticsHelper:
    """Utility class for statistical calculations"""
    
    @staticmethod
    def calculate_mean(values: List[Union[int, float]]) -> float:
        """Calculate mean of values"""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def calculate_median(values: List[Union[int, float]]) -> float:
        """Calculate median of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    @staticmethod
    def calculate_std_deviation(values: List[Union[int, float]]) -> float:
        """Calculate standard deviation of values"""
        if len(values) <= 1:
            return 0.0
        
        mean = StatisticsHelper.calculate_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    @staticmethod
    def calculate_percentile(values: List[Union[int, float]], percentile: float) -> float:
        """Calculate specified percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            if upper_index >= len(sorted_values):
                return sorted_values[-1]
            
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


class ErrorHandler:
    """Utility class for error handling and logging"""
    
    @staticmethod
    def log_error(error: Exception, context: str = "") -> str:
        """Log error and return error ID"""
        error_id = str(uuid.uuid4())[:8]
        error_msg = f"[{error_id}] {context}: {str(error)}"
        
        logger.error(error_msg, exc_info=True)
        
        return error_id
    
    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {e}")
            return default


# Example usage and testing
if __name__ == "__main__":
    # Test text processing
    sample_text = """
    1. What is artificial intelligence?
    This is the answer to question 1.
    
    2. Explain machine learning.
    Machine learning is a subset of AI that enables computers to learn.
    """
    
    # Test question extraction
    questions = TextProcessor.extract_question_numbers(sample_text)
    print(f"Found questions: {questions}")
    
    # Test question splitting
    split_questions = TextProcessor.split_by_questions(sample_text)
    print(f"Split questions: {split_questions}")
    
    # Test keyword extraction
    keywords = TextProcessor.extract_keywords("This is a sample text for keyword extraction")
    print(f"Keywords: {keywords}")
    
    # Test file operations
    print(f"File size: {FileHandler.format_file_size(1024000)}")
    
    # Test statistics
    values = [85, 92, 78, 96, 88, 75, 90]
    print(f"Mean: {StatisticsHelper.calculate_mean(values):.2f}")
    print(f"Median: {StatisticsHelper.calculate_median(values):.2f}")
    print(f"Std Dev: {StatisticsHelper.calculate_std_deviation(values):.2f}")
