# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_excel('/home/master1/Git/Option-CDEFGH/Option C/Option-c prediction 6/OPTION-C.xlsx')

# %%
df.head()

# %%
df1= df.dropna( how='all')



# %%
# Remove columns that start with "Unnamed"

# Correct code to drop columns by index
columns_to_drop = df1.columns[26:34] # Indices of the columns to drop
df1.drop(columns_to_drop, axis=1, inplace=True)

# Display the first few rows of the updated DataFrame
df1.head()

# %%
df1.columns=["Col"+str(i) for i in range(0, 26)]

# %%
df1.head(3)

# %%
X = df1.drop('Col25',axis=1)
y = df1['Col25']
print(X.shape)
print(y.shape)

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.11, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape) 

# %%
df1.head()

# %%
# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Print the shape of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)


# %%
# Setting the seed

SEED = 42


# %%
# Instatiating decision tree
from sklearn.tree import DecisionTreeClassifier
tree =  DecisionTreeClassifier(max_depth = 5, random_state=SEED)


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# %%
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def print_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

# Assuming you have defined y_test and y_pred earlier

print_scores(y_test, y_pred)


