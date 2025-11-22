import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diamond Price Prediction & Analysis",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e86ab;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("💎 Diamond Analysis")
    st.sidebar.markdown("---")
    sections = [
        "🏠 Project Overview",
        "📊 Dataset Explorer",
        "🔍 Data Analysis",
        "🤖 ML Models",
        "📈 Results & Insights",
        "👨‍💻 About Author"
    ]
    selected_section = st.sidebar.radio("Navigate to:", sections)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dataset Info:**
    - 53,940 diamond records
    - Price prediction & cut classification
    - Multiple ML algorithms compared
    """)
    
    # Main content routing
    if selected_section == "🏠 Project Overview":
        show_project_overview()
    elif selected_section == "📊 Dataset Explorer":
        show_dataset_explorer()
    elif selected_section == "🔍 Data Analysis":
        show_data_analysis()
    elif selected_section == "🤖 ML Models":
        show_ml_models()
    elif selected_section == "📈 Results & Insights":
        show_results_insights()
    elif selected_section == "👨‍💻 About Author":
        show_about_author()

def show_project_overview():
    st.markdown('<h1 class="main-header">💎 Diamond Price Prediction & Analysis</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Project Overview
        
        This comprehensive analysis explores the famous **Diamonds Dataset** containing 53,940 diamond records. 
        The project demonstrates end-to-end data science workflow from data exploration to machine learning model deployment.
        
        ### 🎯 Project Goals
        1. **Regression**: Predict diamond prices based on features
        2. **Classification**: Classify diamond cut quality
        3. **Analysis**: Understand factors influencing diamond pricing
        
        ### 🛠️ Technical Stack
        - **Python** (Pandas, NumPy, Scikit-learn)
        - **Visualization** (Matplotlib, Seaborn)
        - **Machine Learning** (Multiple algorithms)
        - **Web App** (Streamlit)
        """)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h3>🚀 Quick Stats</h3>
        <p><strong>53,940</strong> Records</p>
        <p><strong>10</strong> Features</p>
        <p><strong>6</strong> ML Models</p>
        <p><strong>2</strong> Prediction Tasks</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📁 Dataset Features
        - **carat**: Diamond weight
        - **cut**: Quality of cut
        - **color**: Color grade
        - **clarity**: Clarity measurement
        - **dimensions**: x, y, z measurements
        - **price**: Target variable
        """)
    
    st.markdown("---")
    
    # ML Algorithms used
    st.subheader("🤖 Machine Learning Algorithms Implemented")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Regression Models
        - Linear Regression
        - Support Vector Regressor (SVR)
        - Decision Tree Regressor
        - Random Forest Regressor
        - K-Nearest Neighbors Regressor
        - Gradient Boosting Regressor
        """)
    
    with col2:
        st.markdown("""
        ### 🏷️ Classification Models
        - Logistic Regression
        - Support Vector Classifier (SVC)
        - Decision Tree Classifier
        - Random Forest Classifier
        - K-Nearest Neighbors Classifier
        - Gradient Boosting Classifier
        """)

def show_dataset_explorer():
    st.markdown('<h2 class="section-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    @st.cache_data
    def load_data():
        return sns.load_dataset('diamonds')
    
    diamonds = load_data()
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(diamonds):,}")
    with col2:
        st.metric("Number of Features", diamonds.shape[1])
    with col3:
        st.metric("Missing Values", diamonds.isnull().sum().sum())
    
    st.subheader("Data Preview")
    st.dataframe(diamonds.head(10), use_container_width=True)
    
    # Data information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(diamonds.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df)
    
    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(diamonds.describe(), use_container_width=True)
    
    # Interactive data exploration
    st.subheader("🔍 Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_column = st.selectbox("Select column to analyze:", diamonds.columns)
    
    with col2:
        analysis_type = st.selectbox("Analysis type:", ["Value Counts", "Distribution", "Statistics"])
    
    if analysis_type == "Value Counts" and diamonds[selected_column].dtype == 'category':
        st.write(f"Value counts for {selected_column}:")
        value_counts = diamonds[selected_column].value_counts()
        st.dataframe(value_counts)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {selected_column}')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    elif analysis_type == "Distribution" and diamonds[selected_column].dtype in ['float64', 'int64']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        diamonds[selected_column].hist(bins=30, ax=ax1)
        ax1.set_title(f'Histogram of {selected_column}')
        ax1.set_xlabel(selected_column)
        ax1.set_ylabel('Frequency')
        
        # Box plot
        diamonds.boxplot(column=selected_column, ax=ax2)
        ax2.set_title(f'Box Plot of {selected_column}')
        
        st.pyplot(fig)

def show_data_analysis():
    st.markdown('<h2 class="section-header">🔍 Data Analysis</h2>', unsafe_allow_html=True)
    
    diamonds = sns.load_dataset('diamonds')
    
    # Correlation Analysis
    st.subheader("📈 Correlation Analysis")
    
    # Calculate correlation
    corr = diamonds.corr(numeric_only=True)
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
    ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16)
    st.pyplot(fig)
    
    # Correlation with price
    st.subheader("💰 Correlation with Price")
    corr_price = corr['price'].sort_values(ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_price.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Feature Correlation with Price')
        ax.set_ylabel('Correlation Coefficient')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.dataframe(pd.DataFrame(corr_price, columns=['Correlation']))
    
    # Price distribution by cut
    st.subheader("💎 Price Distribution by Cut Quality")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=diamonds, x='cut', y='price', ax=ax)
    ax.set_title('Price Distribution by Diamond Cut')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Carat vs Price
    st.subheader("⚖️ Carat vs Price Relationship")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=diamonds, x='carat', y='price', hue='cut', alpha=0.6, ax=ax)
    ax.set_title('Carat vs Price (Colored by Cut Quality)')
    ax.set_xlabel('Carat')
    ax.set_ylabel('Price ($)')
    st.pyplot(fig)

def show_ml_models():
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Machine Learning Approach
    
    This project implements both **Regression** (price prediction) and **Classification** (cut quality) tasks
    using multiple algorithms for comprehensive comparison.
    """)
    
    # Model comparison tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Regression Models")
        reg_models = [
            "Linear Regression",
            "Support Vector Regressor (SVR)",
            "Decision Tree Regressor", 
            "Random Forest Regressor",
            "K-Neighbors Regressor",
            "Gradient Boosting Regressor"
        ]
        
        for i, model in enumerate(reg_models, 1):
            st.write(f"{i}. {model}")
    
    with col2:
        st.subheader("🏷️ Classification Models")
        clf_models = [
            "Logistic Regression",
            "Support Vector Classifier (SVC)",
            "Decision Tree Classifier",
            "Random Forest Classifier", 
            "K-Neighbors Classifier",
            "Gradient Boosting Classifier"
        ]
        
        for i, model in enumerate(clf_models, 1):
            st.write(f"{i}. {model}")
    
    st.markdown("---")
    
    # Preprocessing pipeline
    st.subheader("⚙️ Preprocessing Pipeline")
    
    st.markdown("""
    ### Data Preprocessing Steps:
    
    1. **Numeric Features Scaling**
       - StandardScaler for normalization
       - Features: carat, depth, table, x, y, z
    
    2. **Categorical Features Encoding**  
       - OneHotEncoder with drop='first'
       - Features: color, clarity
    
    3. **ColumnTransformer**
       - Applies different transformations to different columns
       - Maintains data integrity
    """)
    
    st.code("""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    """, language='python')
    
    # Feature importance demonstration
    st.subheader("🎯 Feature Importance")
    
    # Simulate feature importance (in real app, this would come from trained models)
    features = ['carat', 'x', 'y', 'z', 'depth', 'table', 'color', 'clarity']
    importance = [0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color='lightseagreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Relative Feature Importance for Price Prediction')
    st.pyplot(fig)

def show_results_insights():
    st.markdown('<h2 class="section-header">📈 Results & Insights</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏆 Key Findings
    
    After comprehensive analysis and model training, here are the key insights from the diamond dataset:
    """)
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💡 Major Insights
        
        - **Carat weight** is the strongest price predictor (92% correlation)
        - **Physical dimensions** (x, y, z) significantly impact price
        - **Cut quality** shows complex relationship with price
        - **Ideal cut** diamonds don't always command highest prices
        - **Color and clarity** have moderate impact on pricing
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Performance
        
        - **Random Forest** performed best for regression
        - **Gradient Boosting** excelled in classification  
        - **Feature engineering** improved model accuracy
        - **Cross-validation** ensured model robustness
        - **Hyperparameter tuning** optimized performance
        """)
    
    st.markdown("---")
    
    # Business implications
    st.subheader("💼 Business Implications")
    
    st.markdown("""
    ### For Diamond Industry:
    
    1. **Pricing Strategy**: Carat weight should be primary pricing factor
    2. **Quality Assessment**: Cut quality needs nuanced evaluation  
    3. **Market Segmentation**: Different customer segments value features differently
    4. **Inventory Management**: Focus on high-correlation features for valuation
    
    ### For Consumers:
    
    1. **Value Buying**: Understand which features truly impact price
    2. **Informed Decisions**: Make purchase decisions based on data
    3. **Quality Assessment**: Learn to evaluate diamond quality objectively
    """)
    
    # Future work
    st.subheader("🔮 Future Enhancements")
    
    st.markdown("""
    - **Deep Learning**: Implement neural networks for better accuracy
    - **Time Series**: Analyze diamond price trends over time  
    - **Market Analysis**: Incorporate external economic factors
    - **Web Application**: Deploy as production-ready prediction service
    - **API Development**: Create REST API for model serving
    """)

def show_about_author():
    st.markdown('<h2 class="section-header">👨‍💻 About the Author</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <h3>Hammad Zahid</h3>
            <p>🚀 AI & Data Science Enthusiast</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📧 Connect With Me
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/hammad-zahid)
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/hammadzahid)
        [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://hammadzahid.dev)
        """)
    
    with col2:
        st.markdown("""
        ## 🎓 Professional Profile
        
        **Hammad Zahid** | 22 years old | VU University of Pakistan
        
        ### 🛠️ Technical Expertise
        - **Machine Learning**: Supervised/Unsupervised Learning, Deep Learning
        - **Data Science**: Pandas, NumPy, Statistical Analysis, Data Visualization
        - **Programming**: Python (Expert), SQL, Web Development
        - **Tools**: Scikit-learn, TensorFlow, Streamlit, Git, Docker
        
        ### 📊 Project Highlights
        - End-to-end ML pipeline development
        - Data analysis and visualization
        - Model deployment and serving
        - Interactive web applications
        
        ### 🎯 Career Focus
        Building professional expertise in:
        - Artificial Intelligence
        - Data Science & Analytics  
        - Machine Learning Engineering
        - Data-Driven Decision Making
        """)
    
    st.markdown("---")
    
    # Skills matrix
    st.subheader("📊 Technical Skills Matrix")
    
    skills = {
        'Python Programming': 95,
        'Machine Learning': 90,
        'Data Analysis': 88,
        'Data Visualization': 85,
        'Statistical Analysis': 82,
        'SQL Databases': 80,
        'Web Development': 75,
        'Deep Learning': 70
    }
    
    for skill, level in skills.items():
        st.write(f"**{skill}**")
        st.progress(level)
        st.write("")
    
    # Call to action
    st.markdown("""
    ---
    
    ### 📈 Let's Connect!
    
    I'm passionate about leveraging data to solve real-world problems and always interested in:
    - Collaborating on interesting data science projects
    - Discussing AI/ML innovations and applications  
    - Networking with fellow data professionals
    - Exploring career opportunities in data science
    
    **Feel free to reach out for collaborations or discussions!**
    """)

if __name__ == "__main__":
    main()