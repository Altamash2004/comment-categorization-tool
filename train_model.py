import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import TextPreprocessor

class CommentClassifier:
    """
    Comment classification model using Logistic Regression
    Uses TF-IDF vectorization for feature extraction
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # TF-IDF Vectorizer with optimal parameters
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Top 3000 most important words
            ngram_range=(1, 2),  # Use single words and word pairs
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8  # Ignore words that appear in >80% of documents
        )
        # Logistic Regression with balanced class weights
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42
        )
        self.categories = None
    
    def load_data(self, filepath='comments_dataset.csv'):
        """Load dataset from CSV"""
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} comments")
        print(f"Categories: {df['category'].unique()}")
        return df
    
    def prepare_data(self, df):
        """Preprocess and split data"""
        print("\nPreprocessing comments...")
        # Preprocess all comments
        df['processed_comment'] = df['comment'].apply(self.preprocessor.preprocess)
        
        # Split into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_comment'],
            df['category'],
            test_size=0.2,
            random_state=42,
            stratify=df['category']  # Maintain category distribution
        )
        
        print(f"Training set: {len(X_train)} comments")
        print(f"Test set: {len(X_test)} comments")
        
        return X_train, X_test, y_train, y_test, df
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nVectorizing text with TF-IDF...")
        # Transform text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print("Training Logistic Regression model...")
        # Train the classifier
        self.model.fit(X_train_tfidf, y_train)
        
        # Store categories for later use
        self.categories = self.model.classes_
        
        print("✓ Model trained successfully!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{'='*60}")
        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"{'='*60}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return accuracy, y_pred
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.categories,
                    yticklabels=self.categories)
        plt.title('Confusion Matrix - Comment Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    
    def predict(self, comment):
        """
        Predict category for a single comment
        
        Args:
            comment: Raw comment text
        
        Returns:
            Predicted category and confidence score
        """
        # Preprocess
        processed = self.preprocessor.preprocess(comment)
        
        # Vectorize
        vectorized = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(vectorized)[0]
        
        # Get probability scores
        probabilities = self.model.predict_proba(vectorized)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def predict_batch(self, comments):
        """
        Predict categories for multiple comments
        
        Args:
            comments: List of raw comment texts
        
        Returns:
            List of predictions and confidences
        """
        # Preprocess all
        processed = [self.preprocessor.preprocess(c) for c in comments]
        
        # Vectorize
        vectorized = self.vectorizer.transform(processed)
        
        # Predict
        predictions = self.model.predict(vectorized)
        probabilities = self.model.predict_proba(vectorized)
        confidences = [max(probs) for probs in probabilities]
        
        return predictions, confidences
    
    def save_model(self, model_path='comment_classifier_model.pkl',
                   vectorizer_path='tfidf_vectorizer.pkl'):
        """Save trained model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.categories, 'categories.pkl')
        print(f"\n✓ Model saved as '{model_path}'")
        print(f"✓ Vectorizer saved as '{vectorizer_path}'")
    
    def load_model(self, model_path='comment_classifier_model.pkl',
                   vectorizer_path='tfidf_vectorizer.pkl'):
        """Load pre-trained model and vectorizer"""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.categories = joblib.load('categories.pkl')
        print("✓ Model loaded successfully!")


def main():
    """Main training pipeline"""
    print("="*60)
    print("COMMENT CATEGORIZATION MODEL TRAINING")
    print("="*60)
    
    # Initialize classifier
    classifier = CommentClassifier()
    
    # Load data
    df = classifier.load_data('comments_dataset.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, df_processed = classifier.prepare_data(df)
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate model
    accuracy, predictions = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Test with example comments
    print("\n" + "="*60)
    print("TESTING WITH EXAMPLE COMMENTS")
    print("="*60)
    
    test_examples = [
        "Amazing work! Loved the animation.",
        "This is trash, quit now.",
        "The animation was okay but the voiceover felt off.",
        "Follow me for followers!",
        "Can you make one on Python programming?",
        "This reminded me of my childhood."
    ]
    
    for comment in test_examples:
        category, confidence = classifier.predict(comment)
        print(f"\nComment: '{comment}'")
        print(f"→ Category: {category} (Confidence: {confidence:.2%})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()