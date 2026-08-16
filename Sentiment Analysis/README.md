# Sentiment Analysis Using BERT

## Project Overview

This project develops a binary sentiment-classification system for movie reviews using Natural Language Processing (NLP).

Two approaches are implemented and compared:

1. **TF-IDF + Logistic Regression** — traditional machine-learning baseline
2. **BERT (`bert-base-uncased`)** — transformer-based deep-learning model

The objective is to determine how a fine-tuned BERT model performs compared with a traditional TF-IDF-based sentiment classifier.

---

## Business / NLP Objective

Sentiment analysis can be used to automatically classify customer or user feedback as positive or negative.

Potential applications include:

* Customer feedback analysis
* Product-review analysis
* Brand monitoring
* Opinion mining
* Customer experience analysis
* Social-media sentiment analysis

In this project, the task is to classify IMDB movie reviews into:

* **0 → Negative**
* **1 → Positive**

---

## Dataset

The project uses the **IMDB movie-review dataset**.

The dataset contains:

* **25,000 training reviews**
* **25,000 test reviews**
* Binary sentiment labels

The test set is balanced:

| Sentiment | Test Samples |
| --------- | -----------: |
| Negative  |       12,500 |
| Positive  |       12,500 |
| **Total** |   **25,000** |

---

## Project Workflow

```text
IMDB Dataset
      ↓
Data Inspection
      ↓
Exploratory Data Analysis
      ↓
Review-Length Analysis
      ↓
TF-IDF Feature Extraction
      ↓
Logistic Regression Baseline
      ↓
BERT Tokenization
      ↓
Train / Validation Split
      ↓
BERT Fine-Tuning
      ↓
Validation Evaluation
      ↓
Final Test Evaluation
      ↓
Confusion Matrix
      ↓
Classification Report
      ↓
TF-IDF vs BERT Comparison
      ↓
Conclusion
```

---

# Exploratory Data Analysis

Initial exploratory analysis was performed to understand the dataset and review-length distribution.

The analysis includes:

* Sentiment-class distribution
* Review-length analysis
* Minimum review length
* Maximum review length
* Review-length histogram

The observed review lengths range from approximately **52 characters to 13,704 characters**.

Review-length analysis is particularly relevant for BERT because transformer models operate on token sequences with a defined maximum sequence length.

---

# Baseline Model

Before implementing BERT, a traditional NLP baseline was developed using:

**TF-IDF + Logistic Regression**

## TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) converts text into numerical feature vectors based on the importance of words within individual reviews and across the dataset.

The TF-IDF representation generated approximately:

**74,704 features**

## Logistic Regression

Logistic Regression was trained using the TF-IDF representation.

The baseline used:

* 20,000 training reviews
* 5,000 test reviews

The baseline achieved approximately:

* **Accuracy: 89%**
* **F1-score: 89%**

This provides a strong traditional-machine-learning benchmark for comparison with BERT.

---

# BERT Model

The transformer model used in this project is:

```text
bert-base-uncased
```

BERT is a pretrained transformer language model capable of learning contextual relationships between words.

Unlike TF-IDF, which represents text using word-level statistical features, BERT processes the sequence of tokens and uses contextual information to perform classification.

---

# BERT Tokenization

The `bert-base-uncased` tokenizer converts raw reviews into BERT-compatible inputs.

The tokenizer produces:

* `input_ids`
* `attention_mask`

Special tokens required by BERT are also added during tokenization.

A maximum sequence length of:

```text
256 tokens
```

was used.

Longer reviews are truncated, while shorter sequences are padded as required.

---

# Train / Validation / Test Split

The original IMDB training set contains 25,000 reviews.

For BERT training, it was divided into:

```text
25,000 original training reviews
          │
          ├── 22,500 → Training
          │
          └── 2,500  → Validation
```

The original 25,000-review test set was kept completely separate.

```text
25,000 original test reviews
          │
          └── Final evaluation
```

Therefore:

| Dataset    | Samples | Purpose                          |
| ---------- | ------: | -------------------------------- |
| Training   |  22,500 | BERT fine-tuning                 |
| Validation |   2,500 | Model evaluation during training |
| Test       |  25,000 | Final evaluation                 |

Keeping the test set untouched provides a more reliable estimate of final model performance.

---

# BERT Sequence Classification

The pretrained BERT model was configured for binary classification.

```text
Model: bert-base-uncased
Number of labels: 2
Maximum sequence length: 256
```

The model was fine-tuned using the Hugging Face `Trainer` framework.

---

# BERT Training

The model was trained for:

```text
2 epochs
```

The training process used:

* Tokenized reviews
* Attention masks
* Sentiment labels
* Training dataset
* Validation dataset
* Evaluation metrics

The model was evaluated after training using the validation set.

---

# Validation Results

The BERT model achieved the following validation results:

| Metric    | Validation |
| --------- | ---------: |
| Loss      |     0.2958 |
| Accuracy  |     0.8828 |
| Precision |     0.9206 |
| Recall    |     0.8393 |
| F1-score  |     0.8781 |

These results were obtained using the separate **2,500-review validation set**.

---

# Final BERT Test Results

After training, the model was evaluated on the untouched **25,000-review IMDB test set**.

| Metric        |       BERT |
| ------------- | ---------: |
| **Loss**      | **0.2806** |
| **Accuracy**  | **0.8856** |
| **Precision** | **0.9074** |
| **Recall**    | **0.8589** |
| **F1-score**  | **0.8825** |

### Final Accuracy

**88.56%**

### Final F1-score

**88.25%**

These are the primary BERT performance results reported for this project.

---

# Classification Report

The final test-set classification report is:

| Class       | Precision |   Recall | F1-score |    Support |
| ----------- | --------: | -------: | -------: | ---------: |
| Negative    |      0.87 |     0.91 |     0.89 |     12,500 |
| Positive    |      0.91 |     0.86 |     0.88 |     12,500 |
| **Overall** |  **0.89** | **0.89** | **0.89** | **25,000** |

## Interpretation

For **Negative** sentiment:

* Precision = 0.87
* Recall = 0.91
* F1-score = 0.89

For **Positive** sentiment:

* Precision = 0.91
* Recall = 0.86
* F1-score = 0.88

The model therefore identifies a slightly higher proportion of actual negative reviews, while positive predictions have slightly higher precision.

---

# Confusion Matrix

A confusion matrix was generated using predictions from the final 25,000-review test set.

The confusion matrix provides a detailed view of:

* True Negatives
* False Positives
* False Negatives
* True Positives

It complements the overall accuracy and F1-score by showing the types of classification errors made by the model.

---

# Model Comparison

The BERT model was compared with the TF-IDF + Logistic Regression baseline.

| Model                        |   Accuracy |  Precision |     Recall |         F1 |
| ---------------------------- | ---------: | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression |      ~0.89 |      ~0.89 |      ~0.89 |      ~0.89 |
| **BERT**                     | **0.8856** | **0.9074** | **0.8589** | **0.8825** |

## Comparison

The current experiment does **not** demonstrate that BERT outperformed the TF-IDF + Logistic Regression baseline in overall accuracy or F1-score.

The traditional Logistic Regression baseline achieved approximately 89% accuracy and F1-score, while BERT achieved:

* Accuracy: 88.56%
* F1-score: 88.25%

However, BERT achieved higher precision in the recorded comparison.

This is an important finding because it demonstrates that a traditional TF-IDF-based approach can remain highly competitive on sentiment classification.

---

# Key Findings

### 1. Strong traditional baseline

TF-IDF + Logistic Regression provided a strong baseline with approximately 89% accuracy.

### 2. BERT achieved strong performance

The fine-tuned BERT model achieved:

* 88.56% accuracy
* 90.74% precision
* 85.89% recall
* 88.25% F1-score

### 3. BERT did not outperform the baseline

The current BERT configuration did not surpass the traditional baseline in overall accuracy or F1-score.

### 4. Model complexity does not guarantee better results

The experiment demonstrates that a more sophisticated transformer architecture does not automatically produce better performance.

Model performance depends on factors such as:

* Data preparation
* Tokenization
* Sequence length
* Hyperparameters
* Training duration
* Learning rate
* Model architecture
* Dataset characteristics

---

# Limitations

The current project has several limitations.

### Sequence Length

BERT was configured with a maximum sequence length of 256 tokens.

Some longer reviews may therefore lose information because of truncation.

### Limited Training

The BERT model was trained for only two epochs.

Additional controlled experiments could determine whether further training improves performance.

### Hyperparameter Tuning

The current BERT configuration was not extensively optimized.

Potential parameters for tuning include:

* Learning rate
* Batch size
* Number of epochs
* Maximum sequence length
* Weight decay
* Warm-up steps

### Baseline Comparison

The baseline and BERT models use different representations and modelling approaches.

A more rigorous experiment could retrain both approaches under a fully controlled evaluation pipeline.

---

# Future Improvements

Potential improvements include:

* BERT hyperparameter tuning
* Learning-rate experimentation
* Batch-size experimentation
* Testing longer sequence lengths
* Training for additional epochs
* Early stopping
* Testing DistilBERT
* Testing RoBERTa
* Error analysis
* Sarcasm analysis
* Mixed-sentiment review analysis
* Probability calibration
* Cross-validation where appropriate
* More detailed misclassification analysis

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Jupyter Notebook

---

# Project Structure

```text
Sentiment-Analysis-BERT/
│
├── notebooks/
│   └── Sentiment_Analysis_BERT.ipynb
│
├── visualizations/
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── bert_confusion_matrix.png
│   └── model_comparison.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

Update the filenames above if your actual notebook or visualization filenames are different.

---

# How to Run

## 1. Clone the repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the project

```bash
cd Sentiment-Analysis-BERT
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the notebook

Open:

```text
notebooks/Sentiment_Analysis_BERT.ipynb
```

Run the cells sequentially.

---

# Conclusion

This project demonstrates an end-to-end Natural Language Processing workflow for binary sentiment classification.

A traditional **TF-IDF + Logistic Regression** model was first developed as a baseline. A pretrained **BERT (`bert-base-uncased`)** model was then fine-tuned for sentiment classification.

The final BERT model achieved:

* **88.56% accuracy**
* **90.74% precision**
* **85.89% recall**
* **88.25% F1-score**

The results show that the traditional TF-IDF + Logistic Regression model remained highly competitive with the current BERT implementation.

The project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* TF-IDF
* Logistic Regression
* BERT tokenization
* Transformer fine-tuning
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch
* Sentiment classification
* Confusion-matrix analysis
* Classification reports
* Model comparison
* NLP model evaluation

