# Titanic-Survival-Predictor
#To predict whether a person will be able to survive the titanic disaster based on inputs provided by the user and the data present in the dataset
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
titanic_data = pd.read_csv('Titanic-Dataset.csv')
# Handle missing values
titanic_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
titanic_data.dropna(subset=['Embarked'], inplace=True)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])  # Fit and transform on training data
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])  # Fit and transform on training data

# Select relevant features and target variable
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic_data['Survived']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Get user input
pclass = int(input("Enter passenger class (1, 2, 3): "))
sex = input("Enter passenger gender (0 for female, 1 for male): ")
age = int(input("Enter passenger age: "))
sibsp = int(input("Enter the number of siblings/spouses aboard: "))
parch = int(input("Enter the number of parents/children aboard: "))
fare = float(input("Enter the fare: "))
embarked = input("Enter the port of embarkation (0 for C, 1 for Q, 2 for S): ")

# Transform the user input for 'Sex' and 'Embarked' to integer values
sex = 0 if sex == 'female' else 1
embarked_mapping = {'c': 0, 'q': 1, 's': 2}
embarked = embarked_mapping.get(embarked, -1)  # Default to -1 if input is not recognized

user_input = {
  'Pclass': [pclass],
  'Sex': [sex],
  'Age': [age],
  'SibSp':[sibsp],
  'Parch': [parch],
  'Fare': [fare],
  'Embarked': [embarked]
}

# Create a DataFrame from user input
user_data = pd.DataFrame(user_input)

# Standardize user input
user_data = scaler.transform(user_data)

# Make a survival prediction
survival_prediction = clf.predict(user_data)

if survival_prediction[0] == 1:
    print("The person is likely to survive.")
else:
    print("The person is unlikely to survive.")

# Calculate accuracy and display classification report
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
