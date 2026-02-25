# 🧬 Breast Cancer Subtype Predictor

An end-to-end machine learning pipeline built to accurately classify breast cancer samples into their respective subtypes based on clinical and biological features. 

The core objective of this project is to predict the **`CANCER_TYPE_DETAILED`** label using a highly optimized **LightGBM Classifier**. Moving beyond standard exploratory notebooks, this repository is architected with a custom, object-oriented preprocessing and prediction pipeline designed for robust model inference and deployment.

---

## 📊 Key Results & Model Performance
The final LightGBM model was carefully tuned to handle class imbalances and heterogeneous medical data, yielding exceptional performance across major cancer subtypes (such as Invasive Ductal Carcinoma).
| Metric | Score | Note |
| :--- | :--- | :--- |
| **Overall Accuracy** | 99.0% | Near-perfect classification on majority classes. |
| **Macro F1-Score** | 0.931 | Excellent predictive stability across all subtypes. |
| **Balanced Accuracy** | 0.892 | Proves effective handling of imbalanced clinical data. |

🩺 Example Subtypes:
Breast
Breast Invasive Ductal Carcinoma
Breast Invasive Lobular Carcinoma
Breast Invasive Mixed Mucinous Carcinoma
Breast Mixed Ductal and Lobular Carcinoma
Invasive Breast Carcinoma

⚙️ Development Workflow
1. Exploratory Data Analysis (EDA)
Inspected clinical and patient data to understand feature distributions and modality availability flags.

Analyzed missing values, correlations, and class imbalances.

Leveraged visualizations to gain interpretability insights into feature importance for clinical decision-making.

2. Custom Preprocessing & Feature Engineering
Built a custom CancerSubtypeClassifier class (src/pipeline.py) to handle missing values and scale features dynamically.

Utilized LightGBM's native categorical support combined with custom SimpleImputer and LabelEncoder steps.

Successfully merged clinical sample and patient data into a unified, deployment-ready structure.

3. Model Training & Evaluation
Tested multiple baseline models (Logistic Regression, Random Forest, etc.) before selecting LightGBM for its efficiency with heterogeneous data and built-in regularization.

Tuned hyperparameters to achieve the optimal balance between performance and computation time. Focus was placed on balanced metrics suitable for medical classification.

## 📂 Project Architecture

The repository is structured following software engineering best practices for machine learning projects:

```text
Breast-Cancer-Subtype-Predictor/
│
├── data/                   # Raw and processed datasets
│   └── Dataset Of LightGBM.csv
├── models/                 # Serialized model and pipeline components
│   └── lightgbm_model.pkl
├── notebooks/              # Sequential Jupyter notebooks for research and EDA
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_baselines.ipynb
│   └── 03_lightgbm_training.ipynb
├── src/                    # Source code for the production pipeline
│   └── pipeline.py         # Custom OOP class (CancerSubtypeClassifier) for inference
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
⚙️ Development Workflow
1. Exploratory Data Analysis (EDA)
*Inspected clinical and patient data to understand feature distributions and modality availability flags.

*Analyzed missing values, correlations, and class imbalances.

*Leveraged visualizations to gain interpretability insights into feature importance for clinical decision-making.

2. Custom Preprocessing & Feature Engineering
*Built a custom CancerSubtypeClassifier class (src/pipeline.py) to handle missing values and scale features dynamically.

*Utilized LightGBM's native categorical support combined with custom SimpleImputer and LabelEncoder steps.

*Successfully merged clinical sample and patient data into a unified, deployment-ready structure.

3. Model Training & Evaluation
* Tested multiple baseline models (Logistic Regression, Random Forest, etc.) before selecting LightGBM for its efficiency with heterogeneous data and built-in regularization.

* Tuned hyperparameters to achieve the optimal balance between performance and computation time. Focus was placed on balanced metrics suitable for medical classification.

💻 Installation & Usage
1. **Clone the repository and set up a virtual environment:**
```bash
git clone [https://github.com/zeuspavilion/Breast-Cancer-Subtype-Predictor.git](https://github.com/zeuspavilion/Breast-Cancer-Subtype-Predictor.git)
cd Breast-Cancer-Subtype-Predictor
python -m venv venv
```

2. Activate the virtual environment:
```bash
Windows: .\venv\Scripts\activate

Mac/Linux: source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Inference Pipeline:**
You can easily load the pre-trained pipeline in Python to make predictions on new patient data using the custom src/pipeline.py module.

## **🧩 Tech Stack**
Language: Python

Machine Learning: Scikit-learn, LightGBM

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

## **🚀 Clinical Relevance**
This model demonstrates strong potential for clinical application in oncology. By providing highly accurate and interpretable predictions of breast cancer subtypes, this tool can support medical professionals in treatment decision-making and patient stratification for clinical trials.

"Data exploration guided us. Preprocessing refined us. LightGBM perfected us."


