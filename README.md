
# 🧠 Offensive Language Classification using Multilingual BERT, LSTM & Logistic Regression

This repository provides a comprehensive approach to offensive language detection using traditional machine learning and deep learning models. It uses a multilingual dataset to detect and classify offensive text into multiple binary categories but only evaluates the `toxic` label in the final output.

---

## 📁 Project Structure

```
Repository/
├── task/
│   ├── model1_implementation.ipynb  # Logistic Regression And LSTM Model
│   └── model2_implementation.ipynb  # Multilingual BERT (mBERT) Model
├── requirements.txt                 # Required packages for environment
└── README.md                        # Project overview and instructions                            # Project Documentation (This file)
```

---

## 🎯 Objectives

- Train on **six binary offensive labels**:
  - `toxic`, `abusive`, `vulgar`, `menace`, `offense`, `bigotry`
- Predict **only the `toxic` label** for validation and test sets
- Compare model performance using:
  - Logistic Regression (baseline)
  - LSTM-based model
  - Multilingual BERT

---

## 🧪 Models Used

| Model              | Type            | Notes |
|--------------------|-----------------|-------|
| Logistic Regression| Classical ML    | Used TF-IDF features |
| LSTM               | Deep Learning   | Word embeddings + RNN |
| BERT               | Transformer     | `bert-base-multilingual-cased` pretrained model |

---

## ⚙️ Environment Setup

### 🔧 Requirements

- Python 3.10+
- Conda environment (recommended)
- Jupyter Notebook

### 📦 Setup Steps

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/offensive-language-classifier.git
cd offensive-language-classifier

# Step 2: Create and activate conda environment
conda create -n offensive-nlp python=3.10
conda activate offensive-nlp

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Launch Jupyter Notebook
jupyter notebook
```

> ✅ Make sure you have Jupyter installed. If not:
> `pip install notebook`

---

## 📂 Dataset Format

Ensure you place your dataset files (`train.csv`, `validation.csv`, `test.csv`) in the appropriate path as expected by the notebooks.

### Example of train.csv:
```
text,toxic,abusive,vulgar,menace,offense,bigotry
"This is offensive",1,1,0,0,1,0
```

### Example of validation.csv:
```
text,toxic
"This is okay",0
```

### Example of test.csv:
```
text
"This might be offensive"
```

---

## 🚀 How to Run the Models

1. **Logistic Regression + LSTM**  
   📍 Open `task/model1_implementation.ipynb`  
   - Includes preprocessing, logistic regression, LSTM training, evaluation

2. **Multilingual BERT Model**  
   📍 Open `task/model2_implementation.ipynb`  
   - Fine-tunes `bert-base-multilingual-cased`
   - Only evaluates on `toxic` label

Run each notebook cell sequentially to train and evaluate the models.

---

## 📊 Evaluation Metrics

Evaluation focuses **only on the `toxic` label** and includes:

- Accuracy
- F1-Score
- Precision & Recall
- ROC-AUC Score
- Confusion Matrix
- Visualization of ROC & Confusion Matrix

---



## 🛠️ Tools & Libraries Used

- **Data Handling & Preprocessing**: `pandas`, `numpy`, `langdetect`, `re`, `string`, `nltk`
- **Visualization**: `matplotlib`, `seaborn`
- **NLP & Feature Extraction**: `TfidfVectorizer`, `stopwords`, `WordNetLemmatizer`, `BertTokenizer`
- **Machine Learning**: `LogisticRegression`, `Pipeline`, `train_test_split`, `StandardScaler`, `BaseEstimator`, `TransformerMixin`, `FeatureUnion`
- **Evaluation**: `classification_report`, `accuracy_score`, `roc_auc_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `precision_score`, `recall_score`
- **Deep Learning**: `torch`, `torch.nn`, `torch.optim`, `Dataset`, `DataLoader`, `transformers` (for BERT)
- **Utility**: `warnings`, `tqdm`

## ✅ Best Practices Followed

- GPU-accelerated training (where available)
- Modular notebook structure
- Only `toxic` label is used for final predictions
- Separate validation and test predictions
- Preprocessing: Lowercasing, stopword removal, punctuation removal, stemming/lemmatization

---

## 👨‍💻 Author

**Samrat Abdul Jalil**  
AI/ML Developer | Backend Engineer  

Connect on [LinkedIn](https://linkedin.com) | Explore more on [GitHub](https://github.com/)

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use and modify for academic and research purposes.

---
