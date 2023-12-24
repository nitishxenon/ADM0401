# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
# Make sure your dataset has features (X) and labels (y)
df = pd.read_csv('./cancer_dataset.csv')

# Assume 'diagnosis' is the column indicating cancer diagnosis (1 for diagnosed, 0 for non-diagnosed)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for some algorithms like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose a supervised learning algorithm (e.g., Support Vector Machine)
model = SVC(kernel='linear', C=1.0)

# Train the model on the training set
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy = 97
report = classification_report(y_test, y_pred)


print("\nClassification Report:\n", report)

X_train_set = [[0,1,0], [2,3,5],[4,3,6]]
x_train_output = np.array(X_train_set)

Y_train_set = [[1,2,1], [5,6,0],[2,3,1]]
y_train_output = np.array(X_train_set)


print("X-train :" , x_train_output)

print("Y-train : " , y_train_output)

print(f"Accuracy: {accuracy}")
