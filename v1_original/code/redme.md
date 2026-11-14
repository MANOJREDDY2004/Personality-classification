# Personality Classification Using NLP and KNN

## Project Overview

This project performs personality classification using Natural Language Processing (NLP) and Machine Learning techniques.  
The system predicts whether a person is an **Introvert** or **Extrovert** based on textual data extracted from XML datasets.

The project includes complete dataset preparation, preprocessing, feature extraction, model training, evaluation, and prediction.

---

# Project Workflow

```text
XML Files
   ↓
XML to TXT Conversion
   ↓
TXT to CSV Dataset Creation
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
KNN Classification
   ↓
Personality Prediction
```

---

# Technologies Used

- Python
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- K-Nearest Neighbors (KNN)
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Folder Structure

```text
code/
│
├── create_dataset.py
├── english_dataset.csv
├── english_p_c.ipynb
├── english_p_c.py
├── spanish_dataset.csv
├── spanish_p_c.ipynb
├── spanish_p_c.py
├── xml_to_text.py
├── requirements.txt
└── README.md
```

---

# Files Description

## 1. xml_to_text.py
Converts XML files into TXT files by extracting text content from XML documents.

## 2. create_dataset.py
Reads TXT files and converts them into structured CSV datasets with labels.

## 3. english_p_c.py
Python implementation of English personality classification using TF-IDF and KNN.

## 4. english_p_c.ipynb
Jupyter Notebook version of English personality classification model.

## 5. spanish_p_c.py
Python implementation of Spanish personality classification model.

## 6. spanish_p_c.ipynb
Jupyter Notebook version of Spanish personality classification model.

## 7. english_dataset.csv
Processed English dataset used for model training and testing.

## 8. spanish_dataset.csv
Processed Spanish dataset used for model training and testing.

---

# Features

- XML to text conversion
- Automatic dataset generation
- NLP preprocessing
- TF-IDF feature extraction
- KNN-based personality classification
- Confusion matrix visualization
- Accuracy analysis
- K-value comparison graph
- Custom personality prediction

---

# Installation

## Step 1: Install Python

Download Python from:

https://www.python.org/downloads/

---

## Step 2: Install Required Libraries

Open terminal in VS Code and run:

```bash
pip install -r requirements.txt
```

OR manually install:

```bash
pip install pandas scikit-learn matplotlib numpy jupyter notebook ipykernel
```

---

# How to Run the Project

## Run Python Files

```bash
python english_p_c.py
```

OR

```bash
python spanish_p_c.py
```

---

## Run Jupyter Notebook Files

1. Open VS Code
2. Install Python Extension
3. Install Jupyter Extension
4. Open `.ipynb` file
5. Run notebook cells

---

# Model Evaluation

The model performance is evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix
- K Value vs Accuracy Graph

---

# Sample Output

The project generates:

- Personality prediction
- Accuracy score
- Confusion matrix
- Classification metrics
- Performance graphs

---

# Algorithms Used

## TF-IDF Vectorization
Used to convert textual data into numerical feature vectors.

## K-Nearest Neighbors (KNN)
Used for classification of personality traits.

---

# Applications

- Personality Analysis
- Social Media Analytics
- Behavioral Prediction
- NLP Research
- Data Mining Applications

---


# Conclusion

This project demonstrates the complete NLP and Machine Learning workflow for personality classification using textual data.  
It includes dataset preparation, preprocessing, feature extraction, model training, evaluation, and prediction.
