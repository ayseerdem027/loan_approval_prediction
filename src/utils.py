import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import joblib

class Model:

    def __init__(self, file_path):
        self.file = file_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file)
        except FileNotFoundError:
            logging.error(f"File not found: {self.file}")
            raise
    def preprocess_data(self):
        if self.df is None:
            raise ValueError("Dataframe not loaded, call load_data() first")
        
        target_column = 'loan_status'
        X = self.df.drop(columns=target_column)
        y = self.df[target_column]

        # seperate columns by type
        num_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # build preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', StandardScaler(), num_columns),
                ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_columns)
            ]
        )
        file_path = "data/processed/loan_approval_dataset_preprocessed.pkl"
        joblib.dump(preprocessor, file_path)
        logging.info(f"Preprocessor saved to {file_path}")
        X_transformed = preprocessor.fit_transform(X)
        return X_transformed, y

    def use_model_logistic_regression(self, x, y):
        if x is None or y is None:
            raise ValueError("Data not preprocessed, call preprocess_data() first")
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Classification Report:\n{cr}")
        return model
    
    def save_model(self, model):
        if model is None:
            raise ValueError("Model not trained. Call use_model_logistic_regression() first.")
        model_file_path = 'models/logistic_regression_model.pkl'
        joblib.dump(model, model_file_path)
    
    def main(self):
        self.load_data()
        X, y = self.preprocess_data()
        model = self.use_model_logistic_regression(X, y)
        self.save_model(model)

if __name__ == "__main__":
    file_path = 'data/cleaned/loan_approval_dataset_cleaned.csv'
    model = Model(file_path)
    model.main()