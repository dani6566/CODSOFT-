import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


class TitanicSurvivalPrediction:
    def __init__(self, data):
        """Initialize the class with the dataset."""
        self.data = data
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    
    def preprocess_data(self):
        """Preprocess the dataset with feature engineering and encoding."""
        # Fill missing values in 'Age' with the median grouped by 'Pclass' and 'Sex'
        self.data['Age'] = self.data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

        # Fill missing values in 'Embarked' with the mode
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

        # Encode categorical columns
        self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
        self.data['Embarked'] = self.data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        # Create a new feature 'FamilySize'
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1

        # Extract titles from names
        self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

        # Simplify the title categories
        self.data['Title'] = self.data['Title'].replace(['Mlle', 'Ms'], 'Miss')
        self.data['Title'] = self.data['Title'].replace(['Mme'], 'Mrs')
        self.data['Title'] = self.data['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona', 'Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Rare')

        # Encode 'Title' into numeric categories
        self.data['Title'] = self.data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        self.data['Title'] = self.data['Title'].fillna(0)  # Handle any unexpected missing titles

        # Drop columns with non-numeric or unnecessary data
        self.data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

        # Verify there are no remaining non-numeric columns
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        if not non_numeric_cols.empty:
            print(f"Non-numeric columns remaining: {list(non_numeric_cols)}")
            raise ValueError("Non-numeric columns are still present. Please check preprocessing.")

    def correlation(self):
        # Select only numerical columns
        numerical_df = self.data.select_dtypes(include=['number'])

        # Calculate correlations and plot the heatmap
        correlation_matrix = numerical_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()

    def split_data(self):
        """Split the data into training and testing sets."""
        X = self.data.drop('Survived', axis=1)
        y = self.data['Survived']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        # Define numerical and categorical columns
        numerical_features = ['Age', 'Fare', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']

        # Preprocessing pipelines for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combine preprocessors in a column transformer
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
        # Random Forest Classifier
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

        """Train a Random Forest Classifier."""
        # self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        

    def make_predictions(self):
        """Make predictions on the test set."""
        y_pred = self.model.predict(self.X_test)
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        return y_pred, y_pred_prob

    def evaluate_model(self, y_pred, y_pred_prob):
        """Evaluate the model using various metrics."""
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.2f}\n")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Did not survive', 'Survived'],
                    yticklabels=['Did not survive', 'Survived'])
        plt.title('Confusion Matrix')
        plt.show()

        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        auc = roc_auc_score(self.y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend()
        plt.show()

    def display_predictions(self, y_pred):
        """Display predictions for test samples."""
        for i in range(len(y_pred)):
            print(f"X test: {self.X_test.iloc[i].to_dict()} | Predicted: {y_pred[i]}")


# Example Usage
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../Data/Titanic-Dataset.csv")  

    