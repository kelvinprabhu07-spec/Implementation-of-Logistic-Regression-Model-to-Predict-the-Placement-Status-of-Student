# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Matplotlib and Scikit-learn.
2. Create a dataset containing student CGPA and IQ with placement status.
3. Create and train the Logistic Regression model using the training data.
4. Predict the placement status of a new student and display the result.
## Program:
```py
# Program to implement Logistic Regression Model
# to predict the Placement Status of Student.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Input data (CGPA, IQ)
X = np.array([
    [6.0, 90],
    [6.5, 100],
    [7.0, 110],
    [7.5, 120],
    [8.0, 130],
    [8.5, 140],
    [9.0, 150],
    [5.5, 80],
    [7.2, 115],
    [8.8, 145]
])

# Placement status (0 = Not Placed, 1 = Placed)
Y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X, Y)

# Predict placement status for a new student
new_student = np.array([[8.0, 125]])

prediction = model.predict(new_student)

# Display result
if prediction[0] == 1:
    print("Student Placement Status: Placed")
else:
    print("Student Placement Status: Not Placed")

# Display prediction probability
probability = model.predict_proba(new_student)

print("Placement Probability:", round(probability[0][1] * 100, 2), "%")

# Plot graph
plt.scatter(X[:, 0], Y, color='blue', label='Student Data')

plt.scatter(
    new_student[0][0],
    prediction[0],
    color='red',
    label='New Student'
)

plt.xlabel("CGPA")
plt.ylabel("Placement Status")
plt.title("Logistic Regression - Student Placement")
plt.yticks([0, 1], ['Not Placed', 'Placed'])
plt.legend()
plt.grid()
plt.show()
```

## Output:
~~~
Student Placement Status: Placed
Placement Probability: (Model calculated value) %
~~~
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
