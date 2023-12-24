import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Create a DataFrame
columns = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(data=X, columns=columns)
df['diagnosis'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the dataset to a CSV file
df.to_csv('cancer_dataset.csv', index=False)

# Display the first few rows of the dataset
print(df.head())
