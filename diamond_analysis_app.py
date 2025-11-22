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
    page_title="Diamond Dataset Analysis",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    sections = [
        "Introduction",
        "Dataset Overview",
        "Data Processing",
        "Correlation Analysis",
        "Preprocessing",
        "Data Distribution",
        "About Author"
    ]
    selected_section = st.sidebar.radio("Go to:", sections)
    
    # Main content
    if selected_section == "Introduction":
        show_introduction()
    elif selected_section == "Dataset Overview":
        show_dataset_overview()
    elif selected_section == "Data Processing":
        show_data_processing()
    elif selected_section == "Correlation Analysis":
        show_correlation_analysis()
    elif selected_section == "Preprocessing":
        show_preprocessing()
    elif selected_section == "Data Distribution":
        show_data_distribution()
    elif selected_section == "About Author":
        show_about_author()

def show_introduction():
    st.title("💎 Comprehensive Evaluation of Machine Learning Models on Diamond Dataset")
    st.markdown("---")
    
    st.header("Author: Hammad Zahid")
    
    st.subheader("Self Introduction")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        - **Name**: Hammad Zahid
        - **Age**: 22 years old
        - **Education**: Student at VU University of Pakistan
        - **Fields of Expertise**: 
          - Artificial Intelligence
          - Data Science
          - Data Analysis
          - Data Visualization using Python
        - **Professional Focus**: Building professional profiles in AI/Data Science fields
        - **Content Creation**: Python coding content, ML content for social media
        - **Fitness**: Home workout / bodyweight training enthusiast
        """)
    
    with col2:
        st.info("💡 Passionate about leveraging data to solve real-world problems")
    
    st.subheader("Introduction to the Diamonds Dataset")
    st.markdown("""
    The Diamonds dataset is a popular dataset from the ggplot2 package that contains detailed information 
    about **53,940 diamonds**, including their physical characteristics and pricing. It is commonly used 
    in Data Science and Machine Learning projects to analyze how different diamond features influence 
    price and quality.
    """)
    
    # Dataset columns description
    st.subheader("Dataset Columns Description")
    columns_data = {
        "Column": ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"],
        "Description": [
            "Weight of the diamond (most important factor affecting price)",
            "Quality of the diamond's cut (Fair, Good, Very Good, Premium, Ideal)",
            "Diamond color grade (J = worst → D = best / most colorless)",
            "Measure of how clear/free from inclusions the diamond is",
            "Total depth percentage = 100 * (z / mean(x,y))",
            "Width of top of diamond relative to widest point",
            "Price in US dollars (target variable for regression)",
            "Length of the diamond in mm (longest dimension)",
            "Width of the diamond in mm",
            "Depth of the diamond in mm (actual measured depth)"
        ]
    }
    st.table(pd.DataFrame(columns_data))
    
    st.subheader("Summary Understanding")
    st.markdown("""
    - **carat + dimensions (x, y, z)** = physical size
    - **cut, color, clarity** = quality grading attributes  
    - **depth, table** = geometry proportions
    - **price** = what we are trying to predict (regression use case)
    """)

def show_dataset_overview():
    st.title("📊 Dataset Overview")
    st.markdown("---")
    
    # Load dataset
    @st.cache_data
    def load_data():
        return sns.load_dataset('diamonds')
    
    diamonds = load_data()
    
    st.subheader("First 5 Rows of the Dataset")
    st.dataframe(diamonds.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Number of rows: {diamonds.shape[0]}")
        st.write(f"Number of columns: {diamonds.shape[1]}")
        
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame(diamonds.dtypes, columns=['Data Type']))
    
    with col2:
        st.subheader("Missing Values")
        missing_values = diamonds.isnull().sum()
        st.dataframe(pd.DataFrame(missing_values, columns=['Missing Values']))
        
        st.subheader("Basic Statistics")
        st.dataframe(diamonds.describe(), use_container_width=True)
    
    # Categorical variables analysis
    st.subheader("Categorical Variables Analysis")
    
    categorical_cols = diamonds.select_dtypes(include=['category']).columns
    
    for col in categorical_cols:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{col}** value counts:")
            value_counts = diamonds[col].value_counts()
            st.dataframe(value_counts)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            diamonds[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

def show_data_processing():
    st.title("🔧 Data Processing")
    st.markdown("---")
    
    diamonds = sns.load_dataset('diamonds')
    df = diamonds.copy()
    
    st.subheader("Step 1: Check Missing Values")
    missing_values = diamonds.isnull().sum()
    st.dataframe(pd.DataFrame(missing_values, columns=['Missing Values']))
    st.success("✅ No missing values found in the dataset!")
    
    st.subheader("Step 2: Separate Features and Targets")
    
    # Regression target
    y_reg = df['price']
    
    # Classification target
    y_clf = df['cut']
    
    # Features
    X = df.drop(['price', 'cut'], axis=1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Regression Target", "price")
        st.write(f"Shape: {y_reg.shape}")
    
    with col2:
        st.metric("Classification Target", "cut")
        st.write(f"Shape: {y_clf.shape}")
    
    with col3:
        st.metric("Features", "All except price and cut")
        st.write(f"Shape: {X.shape}")
    
    st.subheader("Step 3: Identify Column Types")
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Columns:**")
        for col in numeric_cols:
            st.write(f"- {col}")
    
    with col2:
        st.write("**Categorical Columns:**")
        for col in categorical_cols:
            st.write(f"- {col}")
        if not categorical_cols:
            st.info("No categorical columns found (all are numeric or category type)")

def show_correlation_analysis():
    st.title("📈 Correlation Analysis")
    st.markdown("---")
    
    diamonds = sns.load_dataset('diamonds')
    
    st.subheader("Correlation Matrix")
    
    # Calculate correlation
    corr = diamonds.corr(numeric_only=True)
    
    # Display correlation with price
    st.write("**Correlation with Price:**")
    corr_price = corr['price'].sort_values(ascending=False)
    st.dataframe(pd.DataFrame(corr_price, columns=['Correlation with Price']))
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    
    st.subheader("Key Insights")
    st.markdown("""
    - **carat** has the strongest positive correlation with price (0.92)
    - **x, y, z** dimensions also show strong positive correlation with price
    - **depth** and **table** show weak correlation with price
    - This suggests that physical size (carat and dimensions) are the main price drivers
    """)

def show_preprocessing():
    st.title("⚙️ Preprocessing Pipeline")
    st.markdown("---")
    
    diamonds = sns.load_dataset('diamonds')
    df = diamonds.copy()
    
    X = df.drop(['price', 'cut'], axis=1)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    st.subheader("Preprocessing Steps")
    
    st.markdown("""
    1. **Numeric Features**: StandardScaler for normalization
    2. **Categorical Features**: OneHotEncoder with drop='first'
    3. **ColumnTransformer**: To apply different transformations to different columns
    """)
    
    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    st.subheader("Preprocessor Configuration")
    st.code("""
    ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    """, language='python')
    
    st.info("💡 This preprocessor will handle both numeric scaling and categorical encoding in one pipeline")

def show_data_distribution():
    st.title("📊 Data Distribution Analysis")
    st.markdown("---")
    
    diamonds = sns.load_dataset('diamonds')
    
    st.subheader("Numerical Features Distribution")
    
    numeric_cols = diamonds.select_dtypes(include=['float64', 'int64']).columns
    
    # Let user select which feature to visualize
    selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        diamonds[selected_feature].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {selected_feature}')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(8, 4))
        diamonds.boxplot(column=selected_feature, ax=ax)
        ax.set_title(f'Box Plot of {selected_feature}')
        st.pyplot(fig)
    
    # Show statistics
    st.subheader(f"Statistics for {selected_feature}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{diamonds[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{diamonds[selected_feature].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{diamonds[selected_feature].std():.2f}")
    with col4:
        st.metric("Range", f"{diamonds[selected_feature].min():.2f} - {diamonds[selected_feature].max():.2f}")

def show_about_author():
    st.title("👨‍💻 About the Author")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/200x200/4B8BBE/FFFFFF?text=HZ", 
                caption="Hammad Zahid", width=200)
    
    with col2:
        st.header("Hammad Zahid")
        st.subheader("AI & Data Science Enthusiast")
        
        st.markdown("""
        ### 🎓 Education & Background
        - **Current**: Student at VU University of Pakistan
        - **Age**: 22 years
        - **Focus**: Artificial Intelligence, Data Science, and Machine Learning
        
        ### 💼 Professional Interests
        - Machine Learning Model Development
        - Data Analysis and Visualization
        - Building professional profiles in AI/Data Science
        - Content creation for Python and ML
        
        ### 🏋️ Personal Interests
        - Home workout and bodyweight training
        - No expensive supplements - pure calisthenics!
        - Creating educational content for social media
        
        ### 🛠️ Technical Skills
        - **Programming**: Python (Expert)
        - **Data Science**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Matplotlib, Seaborn, Plotly
        - **ML Algorithms**: Regression, Classification, Clustering
        - **Tools**: Jupyter, Streamlit, Git
        """)
    
    st.markdown("---")
    st.subheader("📱 Social Media & Content")
    st.markdown("""
    I create engaging Python and Machine Learning content including:
    - Short coding clips and tutorials
    - ML project walkthroughs
    - Data analysis case studies
    - Professional development tips in AI/Data Science
    
    *Follow my journey in the world of Data Science!*
    """)

if __name__ == "__main__":
    main()