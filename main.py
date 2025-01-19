import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class PredictiveMaintenanceModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = None
        self.failure_type_encoder = None
        self.clf_binary = None
        self.clf_multi = None

    def load_and_preprocess_data(self):
        """Load dataset, preprocess data, and return features and targets."""
        # Load dataset
        self.df = pd.read_csv(self.dataset_path)

        # Encode categorical features
        self.failure_type_encoder = LabelEncoder()
        self.df['Failure Type Encoded'] = self.failure_type_encoder.fit_transform(self.df['Failure Type'])

        # Define features and targets
        X = self.df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                     'Torque [Nm]', 'Tool wear [min]']]
        y_binary = self.df['Target']  # Binary classification target
        y_multiclass = self.df['Failure Type Encoded']  # Multiclass classification target

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled_binary, y_resampled_binary = smote.fit_resample(X_scaled, y_binary)
        X_resampled_multiclass, y_resampled_multiclass = smote.fit_resample(X_scaled, y_multiclass)

        return X_resampled_binary, y_resampled_binary, X_resampled_multiclass, y_resampled_multiclass

    def train_models(self):
        """Train binary and multiclass models."""
        # Preprocess data
        X_resampled_binary, y_resampled_binary, X_resampled_multiclass, y_resampled_multiclass = self.load_and_preprocess_data()

        # Split data for binary classification
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            X_resampled_binary, y_resampled_binary, test_size=0.2, random_state=42, stratify=y_resampled_binary
        )

        # Split data for multiclass classification
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X_resampled_multiclass, y_resampled_multiclass, test_size=0.2, random_state=42, stratify=y_resampled_multiclass
        )

        # Train binary classifier
        self.clf_binary = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.clf_binary.fit(X_train_bin, y_train_bin)

        # Train multiclass classifier
        self.clf_multi = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.clf_multi.fit(X_train_multi, y_train_multi)

        # Save models and encoders
        joblib.dump(self.clf_binary, 'binary_model.pkl')
        joblib.dump(self.clf_multi, 'multiclass_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.failure_type_encoder, 'failure_type_encoder.pkl')

    def predict_failure(self, air_temp, process_temp, rotational_speed, torque, tool_wear):
        """Predict failure type based on user inputs."""
        # Load models and encoders
        self.clf_multi = joblib.load('multiclass_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.failure_type_encoder = joblib.load('failure_type_encoder.pkl')

        # Prepare input
        input_data = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear]],
                                  columns=['Air temperature [K]', 'Process temperature [K]',
                                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

        # Scale input data
        input_scaled = self.scaler.transform(input_data)

        # Predict failure type
        predicted_class = self.clf_multi.predict(input_scaled)[0]
        predicted_probabilities = self.clf_multi.predict_proba(input_scaled)[0]
        predicted_class_name = self.failure_type_encoder.inverse_transform([predicted_class])[0]

        # Return results
        return predicted_class_name, predicted_probabilities, self.failure_type_encoder.classes_