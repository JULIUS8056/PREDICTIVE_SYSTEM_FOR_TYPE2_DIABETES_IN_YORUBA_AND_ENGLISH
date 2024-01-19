# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

# %%
data1= pd.read_excel('COMBINED DATASET C_M.xlsx')
print("The file uploaded is ",data1)

# %%
print(data1.head())

# %%
categorical_features=['GENDER','FAMILY HISTORY OF DIABETES (FHD)','FAMILY HISTORY OF HYPERTENSION (FHH)','HISTORY OF EXCESS URINE (HEU)','HISTORY OF EXCESS WATING INTAKE (HEW)','PERFORM REGULAR EXERCISE (REX)','HISTORY OF EXESS FOOD (HEF)','PREVIOUS HISTORY OF DIABETES OF ANY TYPE (PHD)']


# %%
X = data1.drop('OUTCOME', axis=1)
y = data1['OUTCOME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

# %%
model = CatBoostClassifier(cat_features=categorical_features)
model = CatBoostClassifier(cat_features=categorical_features, task_type='GPU')
model = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
    custom_loss=['AUC', 'Accuracy'], depth=8,random_seed=42)


# %%
#model.fit(X_train, y_train)

# %%
model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test),
    verbose=False)
#prediction = model.predict(X_test)

# %%



