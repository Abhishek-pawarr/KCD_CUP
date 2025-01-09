import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# Set title and page configuration
st.set_page_config(page_title="Student Performance Prediction", page_icon=":bar_chart:", layout="wide")
st.title("🎓 Student Performance Prediction")

# Function to load and display data
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv("C:\\Kdd cup 2010\\train.csv",sep='\t')
    return data

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'txt'])
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Preprocess datetime columns and step durations
    data['Step Start Time'] = pd.to_datetime(data['Step Start Time'], format='%Y-%m-%d %H:%M:%S.%f')
    data['Step End Time'] = pd.to_datetime(data['Step End Time'], format='%Y-%m-%d %H:%M:%S.%f')
    data['Step Duration'] = (data['Step End Time'] - data['Step Start Time']).dt.total_seconds()
    data.fillna({'Step Duration': 0}, inplace=True)

    st.subheader("Dataset Overview:")
    st.write(data.head())

    # Sidebar filter
    st.sidebar.header("Filters")
    student_ids = data['Anon Student Id'].unique().tolist()
    selected_id = st.sidebar.multiselect("Select Student ID", options=student_ids)

    # Feature Engineering
    st.subheader("Performing Feature Engineering...")

    data['Time Since Last Step'] = data.groupby('Anon Student Id')['Step Start Time'].diff().dt.total_seconds().fillna(0)
    data['Cumulative Hints'] = data.groupby('Anon Student Id')['Hints'].cumsum()
    data['Cumulative Correct'] = data.groupby('Anon Student Id')['Corrects'].cumsum()
    data['Past Performance'] = data.groupby('Anon Student Id')['Correct First Attempt'].expanding().mean().shift().fillna(0).reset_index(level=0, drop=True)
    data['First Attempt Time'] = (pd.to_datetime(data['First Transaction Time']) - data['Step Start Time']).dt.total_seconds().fillna(0)
    data['Needed Hint'] = data['Hints'].apply(lambda x: 1 if x > 0 else 0)
    data['Incorrect Attempts'] = data['Incorrects']
    data['Attempts per Opportunity'] = data.groupby(['Anon Student Id', 'Opportunity(Default)'])['Row'].cumcount() + 1

    st.subheader("Engineered Features Overview:")
    st.write(data[['Step Duration', 'Time Since Last Step', 'Cumulative Hints', 'Past Performance', 'First Attempt Time', 'Needed Hint', 'Incorrect Attempts', 'Attempts per Opportunity']].head())

    # Correlation matrix
    st.subheader("Correlation Matrix of Features")
    fig, ax = plt.subplots(figsize=(15, 6))
    correlation_matrix = data.select_dtypes(include=np.number).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Fill missing values
    data.fillna({
        'Time Since Last Step': data['Time Since Last Step'].mean(),
        'Cumulative Hints': data['Cumulative Hints'].mean(),
        'Past Performance': data['Past Performance'].mean(),
        'First Attempt Time': data['First Attempt Time'].mean(),
        'Incorrect Attempts': data['Incorrect Attempts'].mean(),
        'Attempts per Opportunity': data['Attempts per Opportunity'].mean(),
        'Needed Hint': data['Needed Hint'].mode()[0]
    }, inplace=True)

    # Select features and target
    selected_features = ['Time Since Last Step', 'Cumulative Hints', 'Past Performance',
                         'First Attempt Time', 'Needed Hint', 'Incorrect Attempts', 'Attempts per Opportunity']
    X = data[selected_features]
    y = data['Correct First Attempt']

    # Visualization: Distribution of Correct First Attempt
    st.subheader("Distribution of Correct First Attempt")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax, palette="Set2")
    st.pyplot(fig)

    # Visualization: Distribution of Step Duration
    st.subheader("Distribution of Step Duration")
    fig = px.histogram(data, x='Step Duration', nbins=50, title="Step Duration Distribution", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig)

    # Visualization: Cumulative Hints vs Cumulative Correct
    st.subheader("Cumulative Hints vs Cumulative Correct")
    fig = px.scatter(data, x='Cumulative Hints', y='Cumulative Correct', color='Correct First Attempt', 
                     labels={"Correct First Attempt": "Correct on First Try"}, title="Hints vs Correct First Try")
    st.plotly_chart(fig)

    # Visualization: Average Time Since Last Step by Student
    st.subheader("Average Time Since Last Step by Student")
    avg_time_per_student = data.groupby('Anon Student Id')['Time Since Last Step'].mean().reset_index()
    fig = px.bar(avg_time_per_student, x='Anon Student Id', y='Time Since Last Step', title="Avg Time Between Steps (Per Student)", 
                 labels={'Anon Student Id': 'Student ID', 'Time Since Last Step': 'Average Time (seconds)'}, color='Time Since Last Step')
    st.plotly_chart(fig)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sample a smaller subset for training if the dataset is large
    if len(X_train) > 1000:  # Only sample if more than 1000 samples
        sample_size = int(len(X_train) * 0.1)  # Use 10% of the data for training
        X_train = X_train.sample(n=sample_size, random_state=42)
        y_train = y_train[X_train.index]

    # Dimensionality Reduction
    pca = PCA(n_components=5)  # Reduce to 5 components
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Model Training
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='linear', random_state=42, max_iter=1000),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
    }

    # Hyperparameter tuning setup for Random Forest and SVM
    param_grid_rf = {
        'n_estimators': [st.sidebar.number_input('Random Forest - Number of Trees:', min_value=10, max_value=500, value=100, step=10)],
        'max_depth': [st.sidebar.selectbox('Random Forest - Max Depth:', [None, 10, 20, 30])],
        'min_samples_split': [st.sidebar.number_input('Random Forest - Min Samples Split:', min_value=2, max_value=20, value=2)]
    }

    param_grid_svc = {
        'C': [st.sidebar.number_input('SVM - Regularization Parameter (C):', min_value=0.01, max_value=100.0, value=1.0, step=0.1)],
        'gamma': [st.sidebar.selectbox('SVM - Kernel Coefficient (gamma):', ['scale', 'auto'])], 
    }

    param_grid_hgb = {
        'max_iter': [st.sidebar.number_input('Hist Gradient Boosting - Max Iterations:', min_value=100, max_value=1000, value=100, step=50)],
        'learning_rate': [st.sidebar.number_input('Hist Gradient Boosting - Learning Rate:', min_value=0.01, max_value=0.5, value=0.1, step=0.01)],
        'max_depth': [st.sidebar.number_input('Hist Gradient Boosting - Max Depth:', min_value=1, max_value=20, value=5, step=1)]
    }

    # Sidebar for model selection and training
    st.sidebar.header("Model Training")
    
    # Variable to store trained models
    trained_models = {}
    best_model_info = {}

    for model_name, model in models.items():
        if st.sidebar.checkbox(f"Train {model_name}"):
            with st.spinner(f'Training {model_name}...'):
                if model_name == 'Random Forest':
                    grid_search = GridSearchCV(model, param_grid_rf, cv=3)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    trained_models[model_name] = best_model
                    best_model_info[model_name] = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }
                elif model_name == 'SVM':
                    grid_search = GridSearchCV(model, param_grid_svc, cv=3)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    trained_models[model_name] = best_model
                    best_model_info[model_name] = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }
                elif model_name == 'Hist Gradient Boosting':
                    grid_search = GridSearchCV(model, param_grid_hgb, cv=3)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    trained_models[model_name] = best_model
                    best_model_info[model_name] = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }

                st.success(f"{model_name} trained successfully!")

                # Display best hyperparameters for each model
                st.write(f"*Best Hyperparameters for {model_name}:*")
                st.json(best_model_info[model_name]['best_params'])

    # Model Comparison
    comparison_results = {}
    
    if trained_models:
        for model_name in trained_models.keys():
            model = trained_models[model_name]
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            comparison_results[model_name] = accuracy

        # Button to compare models
        if st.button("Compare Selected Models"):
            if comparison_results:
                best_model_name = max(comparison_results, key=comparison_results.get)
                best_model_accuracy = comparison_results[best_model_name]
                st.success(f"The best model is *{best_model_name}* with an accuracy of *{best_model_accuracy:.4f}*.")
                
                # Show comparison results in a more visible way
                st.subheader("Model Comparison Results")
                comparison_df = pd.DataFrame(comparison_results.items(), columns=['Model', 'Accuracy'])
                st.write(comparison_df)

                # Plot heatmaps for selected models
                for model_name in comparison_results.keys():
                    model = trained_models[model_name]
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues', ax=ax)
                    st.pyplot(fig)

                    # Display the full classification report
                    st.subheader(f"Classification Report for {model_name}")
                    st.text(classification_report(y_test, y_pred))

            else:
                st.warning("Please select at least one model to compare.")
    else:
        st.warning("No models have been trained yet.")