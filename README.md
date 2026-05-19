# 🎫 FUTURE_ML_02 – Support Ticket Classification

**Future Interns | Machine Learning Track | Task 2**

---

## 🎯 Objective
Build an NLP-powered system that automatically classifies customer support tickets into categories and assigns priority levels (High / Medium / Low / Critical), helping support teams respond faster and smarter.

---

## 📁 Folder Structure
```
FUTURE_ML_02/
├── dataset/
│   └── customer_support_tickets.csv    # Real customer support ticket data
├── output/
│   ├── 01_eda_distributions.png
│   ├── 02_type_priority_heatmap.png
│   ├── 03_word_frequency.png
│   ├── 04_confusion_matrix_type.png
│   ├── 05_classifier_comparison.png
│   ├── 06_confusion_matrix_priority.png
│   ├── 07_predicted_distributions.png
│   └── classified_tickets.csv          # Full classified output
├── Ticket_Classification.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
**File:** `dataset/customer_support_tickets.csv`

| Column | Description |
|---|---|
| `Ticket ID` | Unique ticket identifier |
| `Customer Name` | Customer's name |
| `Ticket Type` | Category (Technical issue, Billing inquiry, etc.) |
| `Ticket Subject` | One-line description |
| `Ticket Description` | Full ticket text |
| `Ticket Priority` | Critical / High / Medium / Low |
| `Ticket Status` | Open / Closed / Pending |

- **Rows:** ~8,469
- **Ticket Types:** 5 categories
- **Priority Levels:** 4 levels

---

## 🔧 Key Steps

### 1. Text Cleaning & Tokenization
- Lowercasing
- Removed template placeholders `{product_purchased}`
- Removed URLs, punctuation, digits
- Stopword removal (NLTK)
- Lemmatization (WordNetLemmatizer)
- Combined subject + description for richer features

### 2. Feature Extraction
- **TF-IDF Vectorizer** with 5,000 features
- Unigrams + Bigrams (`ngram_range=(1,2)`)

### 3. Ticket Category Classification
Models trained:
| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| Naive Bayes | — | — |
| Logistic Regression | — | — |
| Random Forest | — | — |

### 4. Priority Tagging
- Logistic Regression classifies into: **Critical / High / Medium / Low**
- Outputs predicted priority for every ticket

### 5. End-to-End Demo
- `classify_ticket(subject, description)` function for real-time prediction
- Full dataset classified and saved to `output/classified_tickets.csv`

---

## 💡 Business Value
- Reduces manual ticket sorting time significantly
- Ensures high-priority tickets are flagged immediately
- Enables data-driven support team resource allocation
- Can be integrated into live helpdesk systems

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook FUTURE_ML_02_Ticket_Classification.ipynb
```

---

## 🛠️ Technologies Used
- Python 3.x
- pandas, numpy
- NLTK (stopwords, lemmatization)
- scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Random Forest)
- matplotlib, seaborn
- Jupyter Notebook

---

## 🔗 GitHub Repository
Repository: `FUTURE_ML_02`  
Track Code: **ML**
