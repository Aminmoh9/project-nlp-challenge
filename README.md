# PROJECT | Natural Language Processing Challenge

## Introduction

This project applies Natural Language Processing (NLP) techniques to classify news articles as real or fake. You will use Python and common data science libraries to build, train, and evaluate models for this task.

## Project Overview

- **Dataset:** Located in `dataset/data.csv` with columns: `label`, `title`, `text`, `subject`, `date`.
- **Goal:** Build a classifier to distinguish real vs fake news.
- **Validation:** Use your model to predict labels for `dataset/validation_data.csv` and generate a new file with predicted labels (0 for fake, 1 for real).

## Project Structure

- `01_EDA.ipynb`: Exploratory Data Analysis
- `02_Preprocessing.ipynb`: Text cleaning and preprocessing
- `03_Feature_Engineering.ipynb`: Feature extraction (TF-IDF, embeddings)
- `04_Model_Training.ipynb`: Model training and comparison
- `05_Advanced_Embeddings.ipynb`: Embedding-based models
- `06_Validation_Prediction.ipynb`: Final predictions and validation
- `config.py`: Configuration variables
- `utils/init.py`: Utility functions
- `dataset/`: Data files

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- xgboost
- sentence-transformers
- matplotlib
- seaborn
- tqdm
- joblib

## Installation

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd project-nlp-challenge
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```
   Or install manually:
   ```sh
   pip install pandas numpy scikit-learn xgboost sentence-transformers matplotlib seaborn tqdm joblib
   ```

## Usage

1. Open Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Run the notebooks in order:
   - 01_EDA.ipynb
   - 02_Preprocessing.ipynb
   - 03_Feature_Engineering.ipynb
   - 04_Model_Training.ipynb
   - 05_Advanced_Embeddings.ipynb
   - 06_Validation_Prediction.ipynb

## Notes

- Do not use the `subject` column for model training to avoid data leakage.
- Ensure reproducibility by running all preprocessing steps and saving models.
- Respect the original format of the validation file when generating predictions.

## Results & Methodology

### Methodology

1. **Exploratory Data Analysis (EDA):**
   - Analyzed data distribution, missing values, and class balance.

2. **Preprocessing:**
   - Cleaned text (lowercasing, removing HTML, contractions, stopwords, lemmatization).
   - Handled missing values and ensured robust text input for models.

3. **Feature Engineering:**
   - Extracted TF-IDF features and sentence embeddings.
   - Combined title and text for richer representation.

4. **Model Training:**
   - Trained and compared Logistic Regression, SVM, Random Forest, XGBoost, Naive Bayes, and embedding-based models.
   - Used GridSearchCV and cross-validation for hyperparameter tuning.


6. **Advanced Embeddings:**
   - Used Sentence Transformers (e.g., all-MiniLM-L6-v2) to generate dense vector representations for combined title and text.
   - Trained and compared models (Logistic Regression, SVM, Random Forest, XGBoost, GaussianNB) on these embeddings.
   - Applied GridSearchCV for hyperparameter tuning and comapred the best performing embedding-based model. However, for the final predictions the best performing traditional model was used due to better F1 score and accuracy.


### Results

- **Best Model:** [XGBoost]
- **Test Accuracy:** [0.9966]
- **Test F1 Score:** [0.9968]

### Key Insights

- Traditional models (e.g., XGBoost) achieved the highest accuracy and F1 score on the test and validation datasets.
- Advanced embedding-based models were explored, but did not outperform the best traditional model in this case.
- Avoided data leakage by excluding the 'subject' feature from model training.
- Robust preprocessing (cleaning, lemmatization, stopword removal) improved model reliability and generalization.
- Hyperparameter tuning and cross-validation were essential for optimal model selection.
### Validation Dataset Results

Performance of the best model on the validation dataset:

| Metric      | Accuracy | Precision | Recall   | F1 Score |
|-------------|----------|-----------|----------|----------|
| Validation  | 0.9840   | 0.9907    | 0.9686   | 1.0000   |    

*Note: These results are for the validation dataset. Previous results above are for the test dataset.*

## Presentation

You can view the project presentation here:

[Project Presentation Link](https://docs.google.com/presentation/d/1qV972UECu2k2oRY_kwEV6DfV4f_VNy-PqK2HJajWpJo/edit?usp=sharing)



