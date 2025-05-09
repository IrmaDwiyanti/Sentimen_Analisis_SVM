# Analisis Sentimen menggunakan SVM

Proyek ini merupakan implementasi analisis sentimen berbasis teks menggunakan algoritma Support Vector Machine (SVM). Proyek dijalankan di platform Google Colab untuk kemudahan komputasi dan akses data.

## 📊 Workflow

### 1. Data Loading & Exploration
- Load `.tsv` file using `pandas`.
- Count number of reviews and group them by sentiment.

### 2. Preprocessing
- **Tokenization**: Splitting text into words.
- **Lemmatization**: Reducing words to their base form.
- **POS Tagging**: Using TextBlob to get parts-of-speech.

### 3. Vectorization
- Use `CountVectorizer` to convert text into numerical bag-of-words.
- Use `TfidfTransformer` to weigh word importance (TF-IDF).

### 4. Model Training
- Split data into train and test sets (80/20).
- Use a pipeline:
  - `CountVectorizer` ➝ `TfidfTransformer` ➝ `SVC`
- Use `GridSearchCV` with cross-validation to find best hyperparameters (`C`, `kernel`, `gamma`).

### 5. Evaluation
- Print best parameters and accuracy.
- Evaluate using `classification_report`, `confusion_matrix`, `f1_score`, `roc_auc_score`.

## 🧠 Algorithms Used

- **Support Vector Machine (SVC)** with:
  - Linear Kernel
  - RBF Kernel

## 📁 Dataset

The dataset used is from Kaggle’s SI650 competition: [https://www.kaggle.com/c/si650winter11/data](https://www.kaggle.com/c/si650winter11/data)

- Format: Tab-Separated Values (`.tsv`)
- Columns:
  - `liked` — sentiment label (`1` = positive, `0` = negative)
  - `text` — the review text

> Upload dataset ke Colab dengan kode berikut:
```python
from google.colab import files
uploaded = files.upload()
