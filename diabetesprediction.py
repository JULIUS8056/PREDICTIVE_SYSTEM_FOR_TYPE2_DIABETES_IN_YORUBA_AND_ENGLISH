import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
import pickle
data= pd.read_excel('diabetes_dataset.xlsx');
print(data.head())
X = data.drop('OUTCOME', axis=1)
y = data['OUTCOME']
data.dropna(subset=['OUTCOME'],inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
model = CatBoostClassifier(cat_features=cat_features)
model = CatBoostClassifier(cat_features=cat_features, task_type='GPU')
model = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
    custom_loss=['AUC', 'Accuracy'])
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    verbose=False)
prediction = model.predict(X_test)
#y_test_f=y_test.astype(float)
#pd.to_numeric(df['column1'], errors='coerce')
print("y_test:",y_test)
print("prediction y:",prediction)
#confusion_matrix(float(y_test), float(prediction))
#print(confusion_matrix)
#model_accuracy = model.score(X_test, y_test)
#print("Model Accuracy:",model_accuracy * 100 , "%")
#pickle.dump(model , open('Type-2-diabetes-prediction.pkl' , 'wb'))
