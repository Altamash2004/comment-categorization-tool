import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
def download_nltk_data():
    """Download necessary NLTK data files"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

class TextPreprocessor:
    """
    Text preprocessing class for comment classification
    Handles cleaning, tokenization, and lemmatization
    """
    
    def __init__(self):
        download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep important words that might indicate category
        # Remove these from stopwords
        keep_words = {'not', 'no', 'but', 'however', 'against', 'down'}
        self.stop_words = self.stop_words - keep_words
    
    def clean_text(self, text):
        """
        Clean the input text
        - Convert to lowercase
        - Remove URLs
        - Remove mentions (@username)
        - Remove hashtags (#tag)
        - Remove special characters (keep some punctuation)
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            # Fallback to simple split if tokenization fails
            return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove stopwords while keeping important negations"""
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens to their base form"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def remove_punctuation(self, tokens):
        """Remove punctuation from tokens"""
        # Keep exclamation marks and question marks as they indicate emotion/questions
        important_punct = {'!', '?'}
        return [word for word in tokens if word not in string.punctuation or word in important_punct]
    
    def preprocess(self, text, return_string=True):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            return_string: If True, return as string; if False, return as list of tokens
        
        Returns:
            Preprocessed text
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(text)
        
        # Step 3: Remove punctuation
        tokens = self.remove_punctuation(tokens)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 5: Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Return as string or list
        if return_string:
            return ' '.join(tokens)
        return tokens
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts to preprocess
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Example usage and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test examples
    test_comments = [
        "Amazing work! Loved the animation.",
        "This is trash, quit now.",
        "The animation was okay but the voiceover felt off.",
        "Follow me for followers! @username #spam",
        "Can you make one on topic X?",
        "This reminded me of my childhood 😢"
    ]
    
    print("Text Preprocessing Examples:\n")
    print("-" * 70)
    
    for comment in test_comments:
        preprocessed = preprocessor.preprocess(comment)
        print(f"Original:     {comment}")
        print(f"Preprocessed: {preprocessed}")
        print("-" * 70)