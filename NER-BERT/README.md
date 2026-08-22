# Named Entity Recognition Using BERT

## Project Overview

This project implements a Named Entity Recognition (NER) system using a pretrained BERT model.

Named Entity Recognition is a Natural Language Processing (NLP) task that identifies and classifies entities in text, such as:

- Person
- Corporation
- Location
- Product
- Group
- Creative-work

The project uses `bert-base-cased` with a token-classification architecture to predict an NER label for each token.

---

## Objective

The objective of this project is to develop and evaluate a transformer-based NER model capable of identifying named entities in previously unseen text.

The project covers the complete NER workflow:

- CoNLL dataset processing
- NER label encoding
- BERT tokenization
- Token-label alignment
- Dataset preparation
- BERT fine-tuning
- Validation
- Test evaluation
- Entity-level performance analysis
- Custom-text inference

---

## Dataset

The project uses CoNLL-formatted Named Entity Recognition datasets.

The datasets contain sequences of tokens with corresponding NER labels.

A simplified example is:

| Token | NER Label |
|---|---|
| John | B-person |
| works | O |
| at | O |
| Google | B-corporation |

The `O` label represents tokens that do not belong to a named entity.

---

## CoNLL Data Processing

The CoNLL files are parsed into sentence-level token and label sequences.

The preprocessing workflow includes:

1. Reading the CoNLL-formatted files.
2. Separating tokens and labels.
3. Grouping tokens into sentences.
4. Creating structured datasets.
5. Encoding NER labels as numerical IDs.

Two mappings are used:

- `label2id` — converts labels into numerical IDs.
- `id2label` — converts numerical predictions back into readable labels.

---

## BERT Model

The pretrained model used in this project is:

```text
bert-base-cased
