# Hate Speech Detection using Logistic Regression

## 📘 Overview

This project focuses on detecting **hate speech** in tweets using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. It involves text preprocessing, visualization, feature extraction with TF-IDF, and model building using Logistic Regression.

---

## ⚙️ Technologies Used

* **Python** 🐍
* **Libraries:** pandas, numpy, seaborn, matplotlib, nltk, sklearn, wordcloud
* **Model:** Logistic Regression
* **Vectorization:** TF-IDF Vectorizer

---

## 📂 Dataset

* **File Used:** `hateDetection_train.csv`
* **Columns:**

  * `tweet`: The tweet text
  * `label`: Target variable (`0` = non-hate, `1` = hate)

---

## 🧹 Data Preprocessing

The preprocessing steps include:

1. **Lowercasing text**
2. **Removing URLs, mentions, hashtags, and punctuation**
3. **Tokenization**
4. **Stopword removal**
5. **Lemmatization** (using NLTK's `WordNetLemmatizer`)
6. **Dropping duplicate tweets**

---

## 📊 Data Visualization

The code provides visual insights into the dataset:

* **Class distribution** (via countplot and pie chart)
* **Word clouds** for both hate and non-hate tweets to highlight frequent terms.

---

## 🧠 Feature Extraction

* Used **TF-IDF Vectorizer** to convert text into numerical features.
* Configured for **n-grams (1,2)** and **(1,3)** to capture contextual meaning.

---

## 🤖 Model Building

### Model Used:

* **Logistic Regression**

### Process:

1. **Data split:** 80% training, 20% testing using `train_test_split`.
2. **Training:** Logistic Regression model fitted on the training set.
3. **Evaluation:** Model performance measured using accuracy, confusion matrix, and classification report.

### Example Output:

```
Test accuracy: 90.23%
Precision, Recall, F1-Score displayed per class
```

---

## 🔍 Model Optimization

Used **GridSearchCV** to tune hyperparameters:

* `C`: [100, 10, 1.0, 0.1, 0.01]
* `solver`: ['newton-cg', 'lbfgs', 'liblinear']

The grid search identifies the best parameters for optimal model accuracy.

---

## 📈 Results

* **Final Accuracy:** ~90% (after parameter tuning)
* **Confusion Matrix:** Visualized using `ConfusionMatrixDisplay`
* **Classification Report:** Provides precision, recall, and F1-score for both classes.

---

## 🧩 Key Takeaways

✅ Text preprocessing significantly improves accuracy.
✅ TF-IDF with higher n-grams enhances context understanding.
✅ Logistic Regression performs efficiently for binary text classification.
✅ Visualization helps understand bias or imbalance in the dataset.

---

## 🚀 Future Improvements

* Incorporate advanced models like **LSTM**, **BERT**, or **RoBERTa**.
* Implement real-time hate speech detection API.
* Handle multi-class or multilingual hate speech datasets.
