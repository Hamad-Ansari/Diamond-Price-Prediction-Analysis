import streamlit as st
import nbformat
from nbconvert import HTMLExporter

st.set_page_config(page_title="ML Presentation", layout="wide")

st.title("📊 Machine Learning Presentation Viewer")

# -------------------------------------------------------
# LOAD & DISPLAY THE NOTEBOOK AS HTML
# -------------------------------------------------------
st.subheader("📘 Your Presentation")

notebook_path = "Presentation_1.ipynb"    # your file name here

try:
    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Convert to HTML
    html_exporter = HTMLExporter()
    html_data, _ = html_exporter.from_notebook_node(notebook)

    # Display in Streamlit
    st.components.v1.html(html_data, height=800, scrolling=True)

except Exception as e:
    st.error(f"Error loading notebook: {e}")

# -------------------------------------------------------
# ML MODEL EXPLANATION SECTION
# -------------------------------------------------------

st.markdown("---")
st.header("🧠 Which Machine Learning Model is Best?")

st.write("""
Choosing the best ML model depends on:
- The **type of data**
- The **size of data**
- Whether the task is **classification, regression, clustering, or prediction**
- Whether the data is **linear or non-linear**

Below is a simple guide:
""")

st.subheader("🔹 **1. Classification Tasks (Predict Categories)**")
st.write("""
| Problem Type | Best Models | Why |
|--------------|-------------|------|
| Binary classification | Logistic Regression, SVM, Random Forest | Simple, fast, interpretable |
| Multi-class classification | Random Forest, XGBoost, LightGBM | Handles complex decision boundaries |
| High-dimensional data | SVM, Naive Bayes | Works well with many features |
| Image classification | CNN, ResNet, EfficientNet | Deep learning is best for images |
""")

st.subheader("🔹 **2. Regression Tasks (Predict Numbers)**")
st.write("""
| Problem Type | Best Models | Why |
|--------------|-------------|------|
| Linear relationship | Linear Regression, Ridge, Lasso | Simple + interpretable |
| Non-linear relationship | Random Forest, XGBoost, CatBoost | Handles complex patterns |
| Time-series forecasting | ARIMA, LSTM, Prophet | Works with sequential data |
""")

st.subheader("🔹 **3. Clustering Tasks (Discover Groups)**")
st.write("""
| Problem Type | Best Models |
|--------------|-------------|
| Simple clusters | K-Means |
| Arbitrary shaped clusters | DBSCAN |
| Soft clustering | Gaussian Mixture Models |
""")

st.subheader("🔹 **4. Deep Learning Tasks**")
st.write("""
| Data Type | Model |
|-----------|--------|
| Images | CNN |
| Text | LSTM, GRU, Transformer |
| Audio | CNN + RNN |
| Complex large datasets | Deep Neural Networks |
""")

st.success("🎉 Your presentation is fully integrated with Streamlit, and ML concepts are clearly explained!")