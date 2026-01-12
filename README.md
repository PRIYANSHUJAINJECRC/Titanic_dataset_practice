# Titanic Survival Prediction — End-to-End Machine Learning Project

## 📌 Project Overview
The objective of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on demographic, socio-economic, and travel-related features.

This project demonstrates a **complete, professional machine learning workflow**, including data preprocessing, feature engineering, pipeline construction, model training, hyperparameter tuning, model comparison, and result interpretation.

---

## 📂 Dataset
- **Source:** Kaggle Titanic Dataset  
- **Target Variable:** `Survived`  
  - `0` → Did not survive  
  - `1` → Survived  
- **Key Features:**  
  - Age, Sex, Passenger Class (`Pclass`), Fare  
  - Family-related features (`SibSp`, `Parch`)  
  - Embarkation Port (`Embarked`)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights obtained during EDA:
- The dataset is mildly imbalanced, with more non-survivors than survivors.
- Female passengers had a significantly higher survival rate than males.
- Survival probability increased with higher passenger class.
- Children were more likely to survive compared to adults and seniors.
- Passengers traveling alone had lower survival rates.
- Fare distribution was highly right-skewed, motivating logarithmic transformation.

These insights guided feature engineering and modeling decisions.

---

## 🛠 Feature Engineering
To enhance model performance and capture meaningful patterns, the following features were engineered:

- **FamilySize:** Combined `SibSp` and `Parch` to capture the effect of traveling with family.
- **IsAlone:** Binary feature indicating whether a passenger was traveling alone.
- **AgeGroup:** Age binned into categories (Child, Teen, Adult, Senior) to model non-linear survival behavior.
- **FareLog:** Logarithmic transformation of fare to reduce skewness and stabilize variance.

Feature engineering played a crucial role in improving model generalization.

---

## ⚙️ Data Preprocessing
A robust preprocessing pipeline was implemented using `scikit-learn`:

- **Numerical Features:** Scaled using `StandardScaler`
- **Categorical Features:** Encoded using `OneHotEncoder`
- **Pipeline & ColumnTransformer:**  
  - Ensured reproducibility  
  - Prevented data leakage  
  - Enabled fair model comparison  

---

## 🤖 Models Used
The following models were trained and evaluated:

### Logistic Regression
- Used as a baseline due to simplicity and interpretability
- Hyperparameter tuning performed using `GridSearchCV`
- Regularization strength (`C`) optimized using cross-validation

### Random Forest Classifier
- Used to capture non-linear feature interactions
- Evaluated using the same preprocessing pipeline for fair comparison

---

## 📊 Model Evaluation
Models were evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- 5-fold Cross-Validation

Cross-validation was used to ensure performance stability and avoid overfitting.

---

## ✅ Final Model Selection
After feature engineering and hyperparameter tuning:
- Logistic Regression provided stable performance with strong interpretability.
- Random Forest achieved comparable performance by modeling non-linear interactions.

Considering the balance between performance, interpretability, and simplicity, **Logistic Regression was selected as the final model**.

---

## 📈 Key Learnings
- Feature engineering has a greater impact on performance than changing algorithms.
- Pipelines are essential for reproducibility and preventing data leakage.
- Cross-validation is critical for reliable model comparison.
- Model selection should be evidence-based rather than metric-driven.

---

## 🚀 Future Improvements
- Explore advanced ensemble methods such as Gradient Boosting.
- Handle class imbalance explicitly using class weights or resampling techniques.
- Add additional interaction features if more data is available.
- Deploy the trained model as a prediction API.

---

## 🧰 Tech Stack
- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook / Google Colab  

---

## ▶️ How to Run
1. Clone the repository  
2. Install required dependencies  
3. Run the notebook sequentially  

---

## 👤 Author
**Priyanshu Jain**  
Machine Learning Enthusiast  

---

⭐ If you find this project helpful, feel free to star the repository!
