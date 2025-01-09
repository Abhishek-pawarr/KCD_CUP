# 🎓 Student Performance Prediction

This project provides a **Student Performance Prediction** application built with **Streamlit**. The tool leverages machine learning techniques to analyze educational datasets and predict students' performance based on various engineered features.

## 🚀 Features
- **Data Upload & Exploration**: Upload custom datasets and visualize key insights.
- **Feature Engineering**: Automatic generation of features such as time between steps, cumulative hints, and past performance.
- **Visualizations**: Interactive charts and plots using **Plotly** and **Seaborn** for exploring relationships in the data.
- **Dimensionality Reduction**: Integration of **PCA** to reduce feature complexity.
- **Model Training & Comparison**:
  - Train and fine-tune models such as:
    - Random Forest
    - Support Vector Machines (SVM)
    - Histogram-Based Gradient Boosting
  - Hyperparameter tuning using **GridSearchCV**.
- **Model Evaluation**: Compare models using accuracy scores and classification reports.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Streamlit
- Required libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `plotly`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd student-performance-prediction
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ How to Run
1. Launch the Streamlit app:
   ```bash
   streamlit run kdd_streamlit.py
   ```
2. Upload your dataset (CSV or TXT format).
3. Explore the data, perform feature engineering, and visualize key insights.
4. Train and compare machine learning models.
5. Evaluate model performance and view detailed classification reports.

---

## 📊 Dataset
The tool is designed to work with educational datasets, such as the **KDD Cup 2010 dataset**, which contains student performance data. Ensure the dataset includes necessary features like:
- `Step Start Time`
- `Step End Time`
- `Hints`
- `Correct First Attempt`

---

## 🔧 Technical Overview
1. **Feature Engineering**:
   - Time-based features: `Step Duration`, `Time Since Last Step`
   - Aggregated features: `Cumulative Hints`, `Past Performance`
   - Derived features: `Attempts per Opportunity`, `Needed Hint`

2. **Machine Learning Models**:
   - Models trained on engineered features with dimensionality reduction via PCA.

3. **Visualization**:
   - Heatmaps for feature correlations.
   - Histograms and scatter plots for exploring distributions and relationships.

4. **Interactive Hyperparameter Tuning**:
   - User-friendly sliders and dropdowns in the sidebar for hyperparameter configuration.

---

## 📈 Example Outputs
- Model comparison results with accuracy scores.
- Interactive visualizations, such as scatter plots of hints vs. correct attempts.
- Classification reports with detailed metrics.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for new features, bug fixes, or documentation updates.

---

## 🖇️ Acknowledgements
- **KDD Cup 2010** for providing the dataset.
- The Streamlit and Scikit-learn communities for their excellent tools and documentation.
