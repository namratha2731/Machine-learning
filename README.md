

## ✨ Core Functionalities & Projects

### 1. Optical Character Recognition (OCR) with Hidden Markov Models (HMMs)

- **Purpose:** An OCR system recognizing text in images where font/size are known, using HMMs to model character dependencies.
- **Key Components:**
  - **Data Representation:** Maps letters in images to HMM observed states, inferring the most likely hidden characters.
  - **Probability Tables:**  
    - *Initial Probabilities:* Probability of a character starting a statement (from training text).  
    - *Transition Probabilities:* Probability of one character following another (character co-occurrence).  
    - *Emission Probabilities:* Likelihood that an observed image window belongs to a given character, often naive Bayes pixel-wise.
  - **Algorithms:**  
    - *Simplified Model:* Uses emission and character probabilities only.  
    - *Variable Elimination:* Sums over possible state transitions for more accurate inference.  
    - *Viterbi Algorithm:* Finds the most probable character sequence for an input image via dynamic programming and backtracking.
  - **Feature Extraction:** Uses Discrete Cosine Transform (DCT) on image windows.
  - **Evaluation:** Assesses predictions via BLEU score.

### 2. Regression Analysis (Lab 4 & Lab 5)

- **Linear Regression:** Models and evaluates relationships, reporting MSE, RMSE, MAPE, R².
- **Data Handling:** Reads, preps, and reshapes data from Excel or CSV for models.

### 3. Classification Algorithms

- **Perceptron:** Implements from scratch with random search hyperparameter tuning.
- **Multi-Layer Perceptron (MLP):** Models with hyperparameter tuning.
- **Logistic Regression:** Demo and assignment code for binary/multiclass tasks.
- **Model Evaluation:** Computes accuracy, precision, recall, and F1-score.

### 4. Data Preprocessing & Similarity Analysis

- **Missing Values:** Replaces or fills using mode or means, handles '?' and NaN.
- **Categorical Encoding:** Label Encoding, One-Hot Encoding.
- **Scaling & Outlier Detection:** MinMaxScaler, StandardScaler, z-score, IQR.
- **Similarity Measures:** Jaccard, Simple Matching Coefficient (SMC), Cosine Similarity—often plotted as heatmaps.

### 5. Clustering (Lab 3)

- **K-Means:** Fits and evaluates clusters with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
- **Optimal K Selection:** Numerous approaches implemented and compared.

### 6. Miscellaneous Python Scripts

- Simple scripts for tasks like matrix multiplication, counting vowels/consonants, and counting common elements.

---

## 💻 Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **hmmlearn**
- **OpenCV (cv2)**
- **SciPy (DCT extraction)**
- **Matplotlib, Seaborn**
- **NLTK (`nltk.translate.bleu_score`)**
- **Joblib**
- **Google Colab** support

---

## 📂 Project Structure

```
machine-learning/
├── A1.ipynb                   # Perceptron with Learning Rate Analysis
├── A2.ipynb                   # Hyperparameter Tuning for Perceptron, MLP
├── A3ipynb                    # Perceptron/related assignment
├── A4.ipynb                   # Data loading/analysis
├── A5.ipynb                   # Data preprocessing, feature engineering
├── A6.py                      # Data cleaning, categorical handling
├── A7.py                      # Scaling, outlier detection
├── A8.ipynb                   # Jaccard/SMC similarity measures
├── A9.ipynb                   # Cosine similarity
├── A10.ipynb                  # Clustering heatmaps
├── Chinese_hmm8.ipynb         # OCR with Chinese characters
├── HMM_new_approach (2).ipynb # Main OCR Project: DCT features, HMMs, BLEU score
├── Question 1.py              # Vowel/consonant counter
├── Question 2.py              # Matrix multiplication
├── Question 3.py              # Count common elements
├── Question 4.py              # Matrix transpose
├── lab 3.ipynb                # K-means, cluster evaluation
├── lab4 (1) final.ipynb       # Regression metrics
├── lab5.ipynb                 # Linear regression
├── lab6_solns.ipynb           # Perceptron, activations
├── lab8_solns.ipynb           # Perceptron/MLP tuning
├── prog3 - lab2.py            # Logistic regression
└── Optical-Character-Recognition-using-Hidden-Markov-Models-master.zip/
    ├── README.md
    ├── brown_small.txt
    ├── courier-train.png
    ├── ocr.py
    ├── ocr_solver.py
    └── test-*.png
```

---

## ⚙️ Setup and Installation

1. **Clone the Repository:**
    ```
    git clone 
    cd machine-learning
    ```

2. **Install Dependencies:**
    ```
    pip install numpy pandas scikit-learn hmmlearn opencv-python matplotlib seaborn nltk openpyxl
    ```
    - *Note:* For notebooks, some data files must be placed as directed. Adjust Google Colab paths as needed.
    - If you see NLTK-related errors, install missing datasets:
      ```
      import nltk
      nltk.download('punkt')
      ```

---

## ▶️ Usage

- Most projects are in Jupyter notebooks (`.ipynb`). Open with Jupyter Lab/Notebook or Google Colab and run cell by cell.
- Python scripts (`.py`) are run via terminal:
    ```
    python your_script.py
    ```

**For OCR specifically:**
- From terminal:
    ```
    python ocr.py train-image-file.png train-text.txt test-image-file.png
    ```

---

## 🧠 Key Concepts & Algorithms

- **Supervised Learning:**  
  - Classification: Perceptron, MLP, Logistic Regression  
  - Regression: Linear Regression
- **Unsupervised Learning:**  
  - Clustering: K-Means
- **Sequence Modeling:**  
  - Hidden Markov Models (HMMs) for sequence prediction
- **Feature Engineering:**  
  - Discrete Cosine Transform (DCT) for images  
  - Similarities: Jaccard, SMC, Cosine
- **Data Handling:**  
  - Imputation, encoding, scaling, outlier detection
- **Model Evaluation:**  
  - Regression: MSE, RMSE, MAPE, R²  
  - Classification: Accuracy, Precision, Recall, F1  
  - Sequence: BLEU Score
- **Hyperparameter Tuning:**  
  - Automated via RandomizedSearchCV

