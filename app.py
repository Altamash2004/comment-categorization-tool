import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from train_model import CommentClassifier
from response_generator import ResponseGenerator
import json
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Comment Categorization Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'response_gen' not in st.session_state:
    st.session_state.response_gen = ResponseGenerator()
if 'results' not in st.session_state:
    st.session_state.results = None

def load_model():
    """Load the trained model"""
    try:
        classifier = CommentClassifier()
        classifier.load_model()
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running: python train_model.py")
        return None

def get_category_color(category):
    """Get color for each category"""
    colors = {
        'Praise': '#28a745',
        'Support': '#17a2b8',
        'Constructive Criticism': '#ffc107',
        'Hate': '#dc3545',
        'Threat': '#6f42c1',
        'Emotional': '#fd7e14',
        'Spam': '#6c757d',
        'Question': '#007bff'
    }
    return colors.get(category, '#6c757d')

def create_pie_chart(df):
    """Create pie chart of category distribution"""
    category_counts = df['category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Comment Category Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_bar_chart(df):
    """Create bar chart of category distribution"""
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    fig = px.bar(
        category_counts,
        x='Category',
        y='Count',
        title="Comment Count by Category",
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Comments",
        showlegend=False,
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">💬 Comment Categorization & Reply Assistant</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Analyze comments with AI • Get smart response suggestions • Improve engagement
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Choose a feature:",
            ["🏠 Home", "📝 Single Comment", "📁 Batch Upload", "📊 Analytics", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Status")
        
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                st.session_state.classifier = load_model()
                if st.session_state.classifier:
                    st.success("✅ Model loaded!")
        
        if st.session_state.classifier:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model not loaded")
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if st.session_state.results is not None:
            st.metric("Total Comments", len(st.session_state.results))
            st.metric("Categories", st.session_state.results['category'].nunique())
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Single Comment":
        show_single_comment_page()
    elif page == "📁 Batch Upload":
        show_batch_upload_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Home page with overview"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; color: white;'>
            <h3>📝 Single Analysis</h3>
            <p>Classify individual comments and get instant response suggestions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 10px; color: white;'>
            <h3>📁 Batch Processing</h3>
            <p>Upload CSV/JSON files and process multiple comments at once</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 10px; color: white;'>
            <h3>📊 Analytics</h3>
            <p>Visualize comment distribution and gain insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Supported Categories")
    
    categories = {
        "✅ Praise": "Positive feedback and appreciation",
        "💪 Support": "Encouragement and motivation",
        "💡 Constructive Criticism": "Helpful feedback with improvement suggestions",
        "😠 Hate": "Negative and abusive comments",
        "⚠️ Threat": "Threatening or harmful content",
        "💗 Emotional": "Personal and emotionally resonant comments",
        "🚫 Spam": "Promotional and irrelevant content",
        "❓ Question": "Questions and information requests"
    }
    
    col1, col2 = st.columns(2)
    
    items = list(categories.items())
    mid = len(items) // 2
    
    with col1:
        for emoji_cat, desc in items[:mid]:
            st.markdown(f"**{emoji_cat}**")
            st.write(desc)
            st.write("")
    
    with col2:
        for emoji_cat, desc in items[mid:]:
            st.markdown(f"**{emoji_cat}**")
            st.write(desc)
            st.write("")

def show_single_comment_page():
    """Single comment analysis page"""
    st.header("📝 Analyze Single Comment")
    
    if not st.session_state.classifier:
        st.warning("⚠️ Please load the model first using the sidebar button!")
        return
    
    # Text input
    comment = st.text_area(
        "Enter a comment to analyze:",
        placeholder="Type or paste a comment here...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and comment:
        with st.spinner("Analyzing comment..."):
            # Predict
            category, confidence = st.session_state.classifier.predict(comment)
            
            # Get response template
            template = st.session_state.response_gen.get_response_template(category)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Category", category)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Priority", template['priority'])
            
            # Action and tips
            st.markdown(f"### {template['action']}")
            st.info(f"💡 **Tips:** {template['tips']}")
            
            # Suggested responses
            st.markdown("### 💬 Suggested Responses")
            
            for i, response in enumerate(template['suggested_responses'], 1):
                with st.expander(f"Response Option {i}"):
                    st.write(response)
                    if st.button(f"📋 Copy Response {i}", key=f"copy_{i}"):
                        st.code(response)

def show_batch_upload_page():
    """Batch upload and processing page"""
    st.header("📁 Batch Comment Analysis")
    
    if not st.session_state.classifier:
        st.warning("⚠️ Please load the model first using the sidebar button!")
        return
    
    # File upload
    upload_type = st.radio("Choose input method:", ["Upload File", "Paste Text"])
    
    df = None
    
    if upload_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload CSV or JSON file",
            type=['csv', 'json'],
            help="File should have a 'comment' column"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} comments")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:
        text_input = st.text_area(
            "Paste comments (one per line):",
            height=200
        )
        
        if text_input:
            comments = [line.strip() for line in text_input.split('\n') if line.strip()]
            df = pd.DataFrame({'comment': comments})
            st.success(f"✅ Loaded {len(df)} comments")
    
    # Process button
    if df is not None and st.button("🚀 Process All Comments", type="primary"):
        with st.spinner(f"Processing {len(df)} comments..."):
            # Predict all
            predictions, confidences = st.session_state.classifier.predict_batch(df['comment'].tolist())
            
            # Add results to dataframe
            df['category'] = predictions
            df['confidence'] = [f"{c:.2%}" for c in confidences]
            
            # Get actions
            df['action'] = df['category'].apply(
                lambda x: st.session_state.response_gen.get_action_plan(x)
            )
            df['priority'] = df['category'].apply(
                lambda x: st.session_state.response_gen.get_priority(x)
            )
            
            st.session_state.results = df
            
            st.success(f"✅ Processed {len(df)} comments successfully!")
    
    # Display results
    if st.session_state.results is not None:
        st.markdown("---")
        st.subheader("📋 Results")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by category:",
                options=st.session_state.results['category'].unique(),
                default=st.session_state.results['category'].unique()
            )
        
        with col2:
            search = st.text_input("🔍 Search comments:", "")
        
        # Filter dataframe
        filtered_df = st.session_state.results[
            st.session_state.results['category'].isin(selected_categories)
        ]
        
        if search:
            filtered_df = filtered_df[
                filtered_df['comment'].str.contains(search, case=False, na=False)
            ]
        
        # Display
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="comment_analysis_results.csv",
            mime="text/csv"
        )

def show_analytics_page():
    """Analytics and visualization page"""
    st.header("📊 Comment Analytics")
    
    if st.session_state.results is None:
        st.info("💡 Process some comments first to see analytics!")
        return
    
    df = st.session_state.results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", len(df))
    with col2:
        st.metric("Categories", df['category'].nunique())
    with col3:
        most_common = df['category'].mode()[0]
        st.metric("Most Common", most_common)
    with col4:
        high_priority = len(df[df['priority'].isin(['High', 'Very High', 'Critical'])])
        st.metric("High Priority", high_priority)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_pie_chart(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_bar_chart(df), use_container_width=True)
    
    # Category breakdown
    st.markdown("---")
    st.subheader("📈 Category Breakdown")
    
    category_stats = df.groupby('category').agg({
        'comment': 'count',
        'priority': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
    }).reset_index()
    category_stats.columns = ['Category', 'Count', 'Priority']
    category_stats = category_stats.sort_values('Count', ascending=False)
    
    st.dataframe(category_stats, use_container_width=True)

def show_about_page():
    """About page with project information"""
    st.header("ℹ️ About This Tool")
    
    st.markdown("""
    ### 🎯 Purpose
    This tool helps brands and content creators efficiently manage and respond to user comments 
    by automatically categorizing them using Natural Language Processing (NLP).
    
    ### 🤖 How It Works
    1. **Text Preprocessing**: Comments are cleaned and normalized
    2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
    3. **Classification**: Logistic Regression model predicts the category
    4. **Response Generation**: Smart templates suggest appropriate responses
    
    ### 📚 Categories
    The model can classify comments into 8 categories:
    - **Praise**: Positive feedback
    - **Support**: Encouragement
    - **Constructive Criticism**: Helpful improvement suggestions
    - **Hate**: Negative/abusive content
    - **Threat**: Threatening content
    - **Emotional**: Personal/emotional connections
    - **Spam**: Promotional content
    - **Question**: Questions and suggestions
    
    ### 🛠️ Technology Stack
    - **Language**: Python
    - **ML Framework**: scikit-learn
    - **NLP**: NLTK
    - **UI**: Streamlit
    - **Visualization**: Plotly
    
    ### 📖 Usage Guide
    1. **Load Model**: Click "Load Model" in the sidebar
    2. **Single Analysis**: Analyze individual comments
    3. **Batch Processing**: Upload CSV/JSON files or paste multiple comments
    4. **Analytics**: View distribution and insights
    
    ### 🎓 Project Information
    This is a mini-project for Comment Categorization & Reply Assistant Tool assignment.
    
    **Key Features**:
    - ✅ Handles constructive criticism separately from hate
    - ✅ Provides response templates
    - ✅ Visual analytics
    - ✅ Batch processing
    - ✅ Export functionality
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Python, Streamlit, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()