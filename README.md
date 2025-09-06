# Predictive Modeling for Breast Cancer Diagnosis

![Banner](https://user-images.githubusercontent.com/26413235/149544435-01f5a7e3-b3c1-4578-8686-353818e32e85.png)

##  Overview

This project aims to develop a robust machine learning model to accurately classify breast cancer tumors as either **malignant (بدخیم)** or **benign (خوش‌خیم)** based on diagnostic measurements. The primary dataset used is the **Wisconsin Breast Cancer dataset**, a well-known benchmark in the machine learning community.

The entire workflow, from data exploration to final model evaluation, is documented in the accompanying Jupyter Notebook (`Breast_Cancer_Analysis.ipynb`).

---

## 📋 Table of Contents
- [Methodology](#-methodology)
- [Results](#-results)
- [File Structure](#-file-structure)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)

---

## ⚙️ Methodology

The project follows a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):** Initial analysis was performed to understand the data's structure, distribution, and correlations between features. This step helped in identifying key characteristics of the dataset.

2.  **Data Preprocessing:** The data was split into training (80%) and testing (20%) sets. Feature scaling (`StandardScaler`) was applied to normalize the data, which is crucial for distance-based algorithms like SVM and helps with the convergence of Logistic Regression.

3.  **Model Selection & Training:** Three different classification algorithms were evaluated using 5-fold cross-validation:
    * Logistic Regression
    * Support Vector Machine (SVM)
    * Random Forest

4.  **Hyperparameter Tuning:** The best-performing model from the initial evaluation (`Random Forest`) was selected for further optimization using `GridSearchCV` to find the optimal set of hyperparameters.

5.  **Final Evaluation:** The tuned model was then evaluated on the unseen test dataset to measure its real-world performance using metrics like Accuracy, AUC, Precision, and Recall.

---

## 📊 Results

The final model, a tuned **Random Forest Classifier**, achieved excellent performance on the test set.

-   **Test Accuracy:** **97.37%**
-   **Test AUC Score:** **0.998**

### Confusion Matrix
The confusion matrix below shows the model's predictions on the test set. It correctly identified all malignant cases and only misclassified a few benign cases.

![Confusion Matrix](results/confusion_matrix.png)

### ROC Curve
The ROC curve demonstrates the model's outstanding ability to distinguish between the two classes, with an AUC close to 1.

![ROC Curve](results/roc_curve.png)

---

## 📁 File Structure

```
├── final_model.pkl             # Saved final trained model
├── Breast_Cancer_Analysis.ipynb # Main notebook with all the code
├── README.md                   # Project documentation (this file)
├── requirements.txt            # List of required Python libraries
└── results/
    ├── confusion_matrix.png    # Saved confusion matrix plot
    ├── correlation_matrix.png  # Saved correlation heatmap
    ├── metrics.json            # Final performance metrics
    ├── roc_curve.png           # Saved ROC curve plot
    └── target_distribution.png # Plot of target class distribution
```

---

## 🚀 How to Run

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Breast_Cancer_Analysis.ipynb
    ```
---

## 📚 Dependencies

All necessary libraries are listed in the `requirements.txt` file. The primary libraries used are:
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `jupyter`
