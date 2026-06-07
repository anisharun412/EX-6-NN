<H3>NAME : Arunsamy D</H3>
<H3>REGISTER NO : 212224240016</H3>
<H3>EX. NO.6</H3>

<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>

<H3>Program: </H3>

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
```

### Load Dataset

```python
data = pd.read_csv("heart.csv")

data.head()
```

### Check Dataset

```python
print(data.shape)
print()
print(data.info())
print()
print(data.isnull().sum())
```

### Separate Features and Labels

```python
X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values

print("X Shape:", X.shape)

print("y Shape:", y.shape)
```

### Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
```

### Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(X_train[:5])
```

### Create MLP Model

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    max_iter=6000,
    random_state=42
)
```

### Train Model

```python
mlp.fit(X_train, y_train)

print("Training Completed")
```

### Predict Test Data

```python
y_pred = mlp.predict(X_test)

print(y_pred)
```

### Calculate Accuracy

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy =", accuracy * 100, "%")
```

### Plot Error Convergence

```python
plt.plot(mlp.loss_curve_)

plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.title("Error Convergence")

plt.grid(True)

plt.show()
```

<H3>Output:</H3>

## OUTPUT:

### Load Dataset
<img width="1709" height="306" alt="image" src="https://github.com/user-attachments/assets/bc6a8ce5-ee14-40dd-a5ec-dba13af06ed2" />

### Check Dataset
<img width="1400" height="799" alt="image" src="https://github.com/user-attachments/assets/e784bdab-2946-4a27-b254-d5e972c2f781" />

### Separate Features and Labels
<img width="792" height="268" alt="image" src="https://github.com/user-attachments/assets/26e7c341-b065-496f-8401-b0a1ff5e653f" />

### Split Dataset
<img width="868" height="292" alt="image" src="https://github.com/user-attachments/assets/2c38149e-fc97-47aa-a92a-dc05d373cda7" />

### Feature Scaling
<img width="1145" height="457" alt="image" src="https://github.com/user-attachments/assets/fcfa5284-d087-42bf-978c-f864d674b95e" />

### Create MLP Model and Train Model
<img width="1008" height="321" alt="image" src="https://github.com/user-attachments/assets/1102e98f-c4ef-486b-91e5-b12b9106470e" />

### Predict Test Data
<img width="1127" height="263" alt="image" src="https://github.com/user-attachments/assets/631f8b57-8a42-480e-93fa-c00f83aefc88" />

### Calculate Accuracy

<img width="1564" height="136" alt="image" src="https://github.com/user-attachments/assets/1879fbbe-c884-4060-81d7-86c6b627ee89" />

### Plot Error Convergence
<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/b2e9f708-187c-46d8-a40d-3d5a545ba18e" />

<H3>Results:</H3>

Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
