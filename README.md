## Overview
This project is all about predicting Titanic passenger survival using machine learning. 🚢💻

## Data Exploration

- **Dataset Shape:** The dataset contains X rows and Y columns.
- **Preview:** Check out the first 5 rows of the dataset. 👀

## Data Preprocessing

### Handling Categorical Data

Convert gender information to numerical values for analysis. 🔄

```python
income_set = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
```

### Handling Missing Values

Fill in missing age values with the mean age. ⚖️

```python
X.Age = X.Age.fillna(X.Age.mean())
```

### Train-Test Split

Split the data into training and testing sets. 🧩

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
```

## Model Building

### Gaussian Naive Bayes Model

Build a simple Naive Bayes model for predictions. 🤖

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

#### User Input Prediction

Get predictions based on user input. 🤔🔍

```python
pclassNo = int(input("Enter Person's Pclass number: "))
gender = int(input("Enter Person's Gender 0-female 1-male(0 or 1): "))
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: "))
person = [[pclassNo, gender, age, fare]]
result = model.predict(person)
print(result)

if result == 1:
    print("Person might be Survived")
else:
    print("Person might not be Survived")
```

### Random Forest Classifier Model

Try a more complex Random Forest model. 🌲

```python
from sklearn.ensemble import RandomForestClassifier

# Assuming X_train, y_train are your training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Try a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

#### User Input Prediction

Get predictions with the Random Forest model. 🌐🔍

```python
pclassNo = int(input("Enter Person's Pclass number: "))
gender = int(input("Enter Person's Gender 0-female 1-male(0 or 1): "))
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: "))
person = [[pclassNo, gender, age, fare]]
result = rf_model.predict(person)
print(result)

if result == 1:
    print("Person might be Survived")
else:
    print("Person might not be Survived")
```

## Model Evaluation

### Gaussian Naive Bayes Model

Evaluate the Naive Bayes model accuracy. 🎯

```python
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred) * 100))
```

### Random Forest Classifier Model

Check the validation accuracy of the Random Forest model. 📊🎉

```python
y_val_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)
```

## Conclusion

- The Naive Bayes model achieved an accuracy of approximately 77.58%. 📉
- The Random Forest model achieved a validation accuracy of 85.98%. 📈

Feel free to contribute, share feedback, or suggest improvements! 🚀👩‍💻👨‍💻
