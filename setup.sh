#!/bin/bash

echo "🚀 Setting up Diamond Analysis Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download dataset (if needed)
echo "📥 Setting up data..."
python -c "import seaborn as sns; diamonds = sns.load_dataset('diamonds'); diamonds.to_csv('assets/data/diamond_analysis.csv', index=False)"

echo "✅ Setup complete!"
echo "🎯 To run the application: streamlit run app.py"