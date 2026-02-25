import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

class CancerSubtypeClassifier:
    def __init__(self, model, numeric_cols, categorical_cols, label_encoder, numeric_imputer, cat_imputer, category_mappings):
        """
        Initializes the classifier with the trained model and preprocessing components.

        Args:
            model: Trained LightGBM model.
            numeric_cols (list): List of numeric column names.
            categorical_cols (list): List of categorical column names.
            label_encoder (LabelEncoder): Trained LabelEncoder for the target variable.
            numeric_imputer (SimpleImputer): Trained imputer for numeric features.
            cat_imputer (SimpleImputer): Trained imputer for categorical features.
            category_mappings (dict): Mapping of categorical columns to their category levels (from training).
        """
        self.model = model
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.label_encoder = label_encoder
        self.numeric_imputer = numeric_imputer
        self.cat_imputer = cat_imputer
        self.category_mappings = category_mappings

    def preprocess(self, data):
        """
        Applies preprocessing steps to the input data.
        """
        # Ensure input has all required columns
        missing_cols = set(self.numeric_cols + self.categorical_cols) - set(data.columns)
        for col in missing_cols:
            data[col] = np.nan

        # Separate numeric and categorical columns
        data_num = data[self.numeric_cols].copy()
        data_cat = data[self.categorical_cols].copy()

        # Impute missing values
        data_num_imputed = pd.DataFrame(
            self.numeric_imputer.transform(data_num),
            columns=self.numeric_cols, index=data.index
        )
        data_cat_imputed = pd.DataFrame(
            self.cat_imputer.transform(data_cat),
            columns=self.categorical_cols, index=data.index
        )

        # Combine and set categorical data types
        data_processed = pd.concat([data_num_imputed, data_cat_imputed], axis=1)
        for col in self.categorical_cols:
            if col in self.category_mappings:
                data_processed[col] = data_processed[col].astype('category')
                data_processed[col] = data_processed[col].cat.set_categories(self.category_mappings[col])

        return data_processed

    def predict_proba(self, data):
        """
        Predicts class probabilities for the input data.
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame or a dictionary.")

        data_processed = self.preprocess(data)
        return self.model.predict(data_processed) # LightGBM predict returns probabilities directly for multiclass

    def predict(self, data):
        """
        Predicts the class label for the input data.
        """
        proba = self.predict_proba(data)
        predictions = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(predictions)

def save_pipeline(classifier, filename):
    """
    Saves model and preprocessing components to a .pkl file.
    """
    components = {
        'model': classifier.model,
        'numeric_cols': classifier.numeric_cols,
        'categorical_cols': classifier.categorical_cols,
        'label_encoder': classifier.label_encoder,
        'numeric_imputer': classifier.numeric_imputer,
        'cat_imputer': classifier.cat_imputer,
        'category_mappings': classifier.category_mappings
    }
    with open(filename, 'wb') as f:
        pickle.dump(components, f)

def load_pipeline(filename):
    """
    Loads model and preprocessing components from a .pkl file.
    """
    with open(filename, 'rb') as f:
        components = pickle.load(f)
    return CancerSubtypeClassifier(**components)
