"""
Predicting Drug Resistance in Mycobacterium tuberculosis: A Machine Learning Approach to Genomic Mutation Analysis

by Guillermo Paredes-Gutierrez,Ricardo Perea-Jacobo,Héctor-Gabriel Acosta-Mesa,Efren Mezura-Montes,José Luis Morales Reyes,Roberto Zenteno-Cuevas,Miguel-Ángel Guerrero-Chevannier,Raquel Muñiz-Salazar* and Dora-Luz Flores*

DOI:https://doi.org/10.3390/diagnostics15030279
"""

import itertools
import os
import sys
import time

import joblib
from joblib import dump

import lightgbm
from lightgbm import LGBMClassifier, plot_importance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
import sklearn.metrics
import sklearn.metrics as skm
from sklearn import metrics, preprocessing, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve, RocCurveDisplay
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, GroupKFold, KFold,
    StratifiedGroupKFold, StratifiedKFold, train_test_split
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import tensorflow
from tensorflow import keras

import xgboost
from xgboost import XGBClassifier, plot_importance

import pkg_resources

libs = [
    'itertools',  # estándar, no se necesita en requirements
    'os', 'sys', 'time',  # estándar, no se necesita en requirements
    'joblib',
    'lightgbm',
    'matplotlib',
    'numpy',
    'pandas',
    'seaborn',
    'scikit-learn',
    'tensorflow',
    'xgboost'
]

for lib in libs:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib} no está instalado en el entorno actual.")


# Define the base directory (the folder where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the data directory inside the project folder
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load datasets
print("\nLoading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "VARIANTS.csv"))
df1 = pd.read_csv(os.path.join(DATA_DIR, "Resistencias_SR.csv"))
print("\nDatos de variantes:")
print(df.head())
print("\nDatos de resistencias:")
print(df1.head())

label = pd.DataFrame(df, columns=[ 'UNIQUEID'])
print(label)

label.duplicated()

sample_list = label[~ label.duplicated('UNIQUEID')]
print(sample_list)

data = pd.DataFrame(df, columns=[ 'GENOME_INDEX'])

print(data)

data[~ data.duplicated()]

SNPs = data[~ data.duplicated()]
print(SNPs)

Full_sample = pd.DataFrame(df, columns=[ 'UNIQUEID','GENOME_INDEX'])

# Dataframe

Full_RS = pd.DataFrame(df1)
Full_RS = Full_RS.iloc[:, 0:3]
print(Full_RS)

Full_sample

Data_sample = Full_sample.groupby(["UNIQUEID", "GENOME_INDEX"]).size().unstack(level=-1).fillna(0).reset_index()

print(Data_sample)

datasetRS = pd.concat([Full_RS, Data_sample], axis=1) # 0 = S / 1 = R
datasetRS = datasetRS.drop('UNIQUEID', axis=1).reset_index(drop=True)
datasetRS = datasetRS.dropna().reset_index(drop=True)
datasetRS = datasetRS.astype(int)
datasetRS = datasetRS.astype({'EMB': 'float32'})
datasetRS = datasetRS.astype({'INH': 'float32'})
datasetRS = datasetRS.astype({'RIF': 'float32'})

print(datasetRS)
print(datasetRS.shape)

count_emb_s = ((datasetRS.iloc[:, 0] == 1) & (datasetRS.iloc[:, 1] == 0) & (datasetRS.iloc[:, 2] == 0)).sum()
count_inh_s = ((datasetRS.iloc[:, 0] == 0) & (datasetRS.iloc[:, 1] == 1) & (datasetRS.iloc[:, 2] == 0)).sum()
count_rif_s = ((datasetRS.iloc[:, 0] == 0) & (datasetRS.iloc[:, 1] == 0) & (datasetRS.iloc[:, 2] == 1)).sum()
count_emb_inh_rif = ((datasetRS.iloc[:, 0] == 1) & (datasetRS.iloc[:, 1] == 1) & (datasetRS.iloc[:, 2] == 1)).sum()
count_emb_inh = ((datasetRS.iloc[:, 0] == 1) & (datasetRS.iloc[:, 1] == 1) & (datasetRS.iloc[:, 2] == 0)).sum()
count_emb_rif = ((datasetRS.iloc[:, 0] == 1) & (datasetRS.iloc[:, 1] == 0) & (datasetRS.iloc[:, 2] == 1)).sum()
count_inh_rif = ((datasetRS.iloc[:, 0] == 0) & (datasetRS.iloc[:, 1] == 1) & (datasetRS.iloc[:, 2] == 1)).sum()
count_nada    = ((datasetRS.iloc[:, 0] == 0) & (datasetRS.iloc[:, 1] == 0) & (datasetRS.iloc[:, 2] == 0)).sum()
count_emb = (datasetRS.iloc[:, 0] == 1).sum()
count_inh = (datasetRS.iloc[:, 1] == 1).sum()
count_rif = (datasetRS.iloc[:, 2] == 1).sum()

print("Aislados resistentes a EMB solamente: " , count_emb_s)
print("Aislados resistentes a INH solamente: " , count_inh_s)
print("Aislados resistentes a RIF solamente: " , count_rif_s)
print("\nAislados resistentes a los 3 farmacos: ", count_emb_inh_rif)
print("\nAislados resistentes a EMB y INH: ", count_emb_inh)
print("Aislados resistentes a EMB y RIF: ", count_emb_rif)
print("Aislados resistentes a INH y RIF: ", count_inh_rif)
print("\nAislados susceptibles a los 3 farmacos: ", count_nada)
print("\nAislados resistentes a EMB: ", count_emb)
print("Aislados resistentes a INH: ", count_inh)
print("Aislados resistentes a RIF: ", count_rif)

etiqueta_negativa = ((datasetRS.iloc[:, 0] == 0) & (datasetRS.iloc[:, 1] == 0) & (datasetRS.iloc[:,2] == 0))
cantidad_0 = np.count_nonzero(datasetRS[etiqueta_negativa] == 1, axis = 1)

etiqueta_positiva_emb = (datasetRS.iloc[:, 0] == 1)
cantidad_1 = np.count_nonzero(datasetRS[etiqueta_positiva_emb] == 1, axis = 1)

etiqueta_positiva_inh = (datasetRS.iloc[:, 1] == 1)
cantidad_2 = np.count_nonzero(datasetRS[etiqueta_positiva_inh] == 1, axis = 1)

etiqueta_positiva_rif = (datasetRS.iloc[:, 2] == 1)
cantidad_3 = np.count_nonzero(datasetRS[etiqueta_positiva_rif] == 1, axis = 1)

print("Media de mutaciones en aislados susceptibles a los 3 farmacos: {:.2f}".format(np.mean(cantidad_0)))
print("Cantidad de aislados con susceptibilidad a los 3 farmacos: ",len(cantidad_0))

print("\nMedia de mutaciones en aislados resistentes a EMB: {:.2f}".format(np.mean(cantidad_1)))
print("Cantidad de aislados con resistencia a EMB: ",len(cantidad_1))

print("\nMedia de mutaciones en aislados resistentes a INH: {:.2f}".format(np.mean(cantidad_2)))
print("Cantidad de aislados con resistencia a INH: ",len(cantidad_2))

print("\nMedia de mutaciones en aislados resistentes a RIF: {:.2f}".format(np.mean(cantidad_3)))
print("Cantidad de aislados con resistencia a RIF: ",len(cantidad_3))

cantidad_1 = np.count_nonzero(datasetRS == 1, axis = 1)
print("Media de mutaciones entre las 847 muestras y las +79k de posiciones:", np.mean(cantidad_1))

#Training and test datasets

trainr, testr =np.split(datasetRS.sample(frac=1, random_state=42), [int(.80*len(datasetRS))])

print("Shape de conjuntos de train y test respectivamente \n")
print(trainr.shape)
print(testr.shape)
print("\n")

x_trainr = [trainr.iloc[:, 3:]]
y_trainr = [trainr.iloc[:, :3]]
x_testr  = [testr.iloc[:, 3:]]
y_testr  = [testr.iloc[:, :3]]

x_trainr = np.squeeze(x_trainr)
y_trainr = np.squeeze(y_trainr)
x_testr  = np.squeeze(x_testr)
y_testr  = np.squeeze(y_testr)

print('x_trainr', x_trainr.shape)
print('y_trainr', y_trainr.shape)
print('x_testr',  x_testr.shape)
print('y_testr',  y_testr.shape)
print("\n")


"""Feature reduction with PCA""" 

dataset_pca = datasetRS.drop(labels = ["EMB","INH","RIF"], axis = 1)

scaler=StandardScaler()
scaler.fit(dataset_pca)

scaled_data = scaler.transform(dataset_pca)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 1)
labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Variance')
plt.xlabel('Principal Component')
plt.title('Variance by Principal Component')
plt.show()

# Eliminate variables with zero weights
nonzero_weights = np.abs(pca.components_) > 0
filtered_data = scaled_data[:, np.any(nonzero_weights, axis=0)]

count = 0
for i in per_var:
    if i > 0.2:
        count = count + 1

# printing the intersection
print("The numbers greater than 4 : " + str(count))


pca_test = PCA(n_components=470)
pca_test.fit(dataset_pca)

plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")

plt.show()

pca = PCA(n_components = 0.95)
pca.fit(dataset_pca)
print("Cumulative Variances (Percentage):")
print(np.cumsum(pca.explained_variance_ratio_ * 100))
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')

# Make the scree plot
plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")

pca_components = abs(pca.components_)
print(pca_components)

print('Top 5 most important features in each component')
print('===============================================')
for row in range(pca_components.shape[0]):
    # get the indices of the top 5 values in each row
    temp = np.argpartition(-(pca_components[row]), 5)

    # sort the indices in descending order
    indices = temp[np.argsort((-pca_components[row])[temp])][:5]

    # print the top 5 feature names
    print(f'Component {row}: {dataset_pca.columns[indices].to_list()}')

n_PCA_components = 470
pca = PCA(n_components=n_PCA_components)
principalComponents = pca.fit_transform(dataset_pca)

column_names = [f'PC{i+1}' for i in range(n_PCA_components)]
principalDf = pd.DataFrame(data=principalComponents, columns=column_names)

finalDf = pd.concat([principalDf, datasetRS[["EMB", "INH", "RIF"]]], axis=1)
print(finalDf.shape)

trainr_pca, testr_pca =np.split(finalDf.sample(frac=1, random_state=42), [int(.80*len(finalDf))])

X_train_pca = [trainr_pca.iloc[:, :-3]]
y_train_pca = [trainr_pca.iloc[:, :3]]
X_test_pca  = [testr_pca.iloc[:, :-3]]
y_test_pca  = [testr_pca.iloc[:, :3]]

X_train_pca = np.squeeze(X_train_pca)
y_train_pca = np.squeeze(y_trainr)
X_test_pca  = np.squeeze(X_test_pca)
y_test_pca  = np.squeeze(y_testr)

print(X_train_pca.shape)
print(X_test_pca.shape)
print(y_train_pca.shape)
print(y_test_pca.shape)

# Parameters for XGBC and LGBC

params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 15,
            'learning_rate': 0.001,
            'n_estimators':1000,
            'random_state':74,
        }

# XGBC

classifier1 = MultiOutputClassifier(XGBClassifier(objective='binary:logistic', max_depth= 6,alpha= 15,learning_rate= 0.001,n_estimators=2000,random_state=66, scale_pos_weight=4))
classifier1_pca = MultiOutputClassifier(XGBClassifier(objective='binary:logistic', max_depth= 6,alpha= 15,learning_rate= 0.001,n_estimators=2000,random_state=66, scale_pos_weight=4))

# LGBC

classifier2 = MultiOutputClassifier(LGBMClassifier(n_estimators=2000, random_state=66, learning_rate = 0.001, scale_pos_weight=4))
classifier2_pca = MultiOutputClassifier(LGBMClassifier(n_estimators=2000, random_state=66, learning_rate = 0.001, scale_pos_weight=4))

# GradientBoosting

classifier3 = MultiOutputClassifier(GradientBoostingClassifier(n_estimators = 2000, learning_rate=0.001,max_depth=6,random_state=66))
classifier3_pca = MultiOutputClassifier(GradientBoostingClassifier(n_estimators = 2000, learning_rate=0.001,max_depth=6,random_state=66))

# ANN

classifier4 = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100,100,100,3), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate_init=0.001, max_iter=1000, random_state=66 ))
classifier4_pca = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100,100,100,3), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate_init=0.001, max_iter=1000, random_state=66))

print("Training classifier1...")
start_time = time.time()

classifier1.fit(x_trainr, y_trainr)
print("Train score:", classifier1.score(x_trainr, y_trainr))
yhat1 = classifier1.predict(x_testr)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")
print("Test score:", classifier1.score(x_testr, y_testr))


# Training data division in training and validation
x_train, x_val, y_train, y_val = train_test_split(x_trainr, y_trainr, test_size=0.2, random_state=42)

# Storing results
train_scores = []
val_scores = []

# Incremental training and performance monitoring
for i in range(100, len(x_train), 100):
    classifier1.fit(x_train[:i], y_train[:i])
    y_train_pred = classifier1.predict(x_train[:i])
    y_val_pred = classifier1.predict(x_val)

    train_accuracy = accuracy_score(y_train[:i], y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

    print(f"Iteration {i}: Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}")

# Visualization of performance evolution
plt.figure(figsize=(10, 6))
plt.plot(range(100, len(x_train), 100), train_scores, label='Train Accuracy')
plt.plot(range(100, len(x_train), 100), val_scores, label='Validation Accuracy')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('Evolution of Performance during Training')
plt.legend()
plt.show()

fig,ax = plt.subplots(ncols=3,figsize=(20,8))
plot_importance(classifier1.estimators_[0],height=0.5,ax=ax[0],max_num_features=20)
plot_importance(classifier1.estimators_[1],height=0.5,ax=ax[1],max_num_features=20)
plot_importance(classifier1.estimators_[2],height=0.5,ax=ax[2],max_num_features=20)
ax[0].set_xlabel('Decision Trees')
ax[0].set_title('Feature importance for EMB')
ax[0].set_ylabel('Feature')
ax[1].set_xlabel('Decision Trees')
ax[1].set_title('Feature importance for INH')
ax[1].set_ylabel('Feature')
ax[2].set_xlabel('Decision Trees')
ax[2].set_title('Feature importance for RIF')
ax[2].set_ylabel('Feature')
plt.show()

datasetRS.iloc[:,1].name

print(datasetRS.columns[55982])

"""**Arbitrary reduction**"""

#We take the most important positions for each classification, arranging by weight, referring to the number of trees in which it has appeared

importancias1 = classifier1.estimators_[0].get_booster().get_score(importance_type= 'weight')
importancias2 = classifier1.estimators_[1].get_booster().get_score(importance_type= 'weight')
importancias3 = classifier1.estimators_[2].get_booster().get_score(importance_type= 'weight')

importancia_ordenada1 = sorted(importancias1.items(), key=lambda x: x[1], reverse=True)
caracteristicas1 = [x[0] for x in importancia_ordenada1]
print(caracteristicas1)

importancia_ordenada2 = sorted(importancias2.items(), key=lambda x: x[1], reverse=True)
caracteristicas2 = [x[0] for x in importancia_ordenada2]
print(caracteristicas2)

importancia_ordenada3 = sorted(importancias3.items(), key=lambda x: x[1], reverse=True)
caracteristicas3 = [x[0] for x in importancia_ordenada3]
print(caracteristicas3)

# We make a list of the columns that appear in the decision trees

lista_importancias1 = list(sorted(importancias1, key=importancias1.get, reverse=True))
lista_importancias2 = list(sorted(importancias2, key=importancias2.get, reverse=True))
lista_importancias3 = list(sorted(importancias3, key=importancias3.get, reverse=True))

print(len(lista_importancias1))
print(len(lista_importancias2))
print(len(lista_importancias3))

#We join the lists and ignore the repeated positions.

lista_conjuntos = list(set(lista_importancias1 + lista_importancias2 + lista_importancias3))

#We remove the "f" from the names in the list

lista_reduccion = [col.replace("f", "") for col in lista_conjuntos]
lista_reduccion = [int(col) for col in lista_reduccion]

#We create a new dataframe, which only contains the positions of our list.

datasetRS_reduccion = datasetRS.iloc[:, lista_reduccion]

#Let's see the result, so far we have only made a reduction of the variables, considering the positive results of the models.

print(datasetRS_reduccion.shape)

datasetRS_reduccion = pd.concat([datasetRS_reduccion, datasetRS[["EMB","INH","RIF"]]],axis=1)

print(datasetRS_reduccion)

trainr_reduccion, testr_reduccion =np.split(datasetRS_reduccion.sample(frac=1, random_state=42), [int(.80*len(datasetRS_reduccion))])

X_train_reduccion = [trainr_reduccion.iloc[:, :-3]]
y_train_reduccion = [trainr_reduccion.iloc[:, :3]]
X_test_reduccion  = [testr_reduccion.iloc[:, :-3]]
y_test_reduccion  = [testr_reduccion.iloc[:, :3]]

X_train_reduccion = np.squeeze(X_train_reduccion)
y_train_reduccion = np.squeeze(y_trainr)
X_test_reduccion  = np.squeeze(X_test_reduccion)
y_test_reduccion  = np.squeeze(y_testr)

print(X_train_reduccion.shape)
print(X_test_reduccion.shape)
print(y_train_reduccion.shape)
print(y_test_reduccion.shape)

# XGBC

classifier1_reduccion = MultiOutputClassifier(XGBClassifier(objective='binary:logistic', max_depth= 4,alpha= 15,learning_rate= 0.001,n_estimators=2000,random_state=74, scale_pos_weight=4))

start_time = time.time()

classifier1_reduccion.fit(X_train_reduccion, y_train_reduccion)
print(classifier1_reduccion.score(X_train_reduccion, y_train_reduccion))
yhat1_reduccion = classifier1_reduccion.predict(X_test_reduccion)

end_time = time.time()

training_time = end_time - start_time
print(f"El tiempo de entrenamiento fue de {training_time:.2f} segundos.")

print(classifier1_reduccion.score(X_test_reduccion, y_test_reduccion))

fig,ax = plt.subplots(ncols=3,figsize=(20,8))
plot_importance(classifier1_reduccion.estimators_[0],height=0.5,ax=ax[0],max_num_features=20)
plot_importance(classifier1_reduccion.estimators_[1],height=0.5,ax=ax[1],max_num_features=20)
plot_importance(classifier1_reduccion.estimators_[2],height=0.5,ax=ax[2],max_num_features=20)
ax[0].set_xlabel('Decision Trees')
ax[0].set_title('Feature importance for EMB')
ax[0].set_ylabel('Feature')
ax[1].set_xlabel('Decision Trees')
ax[1].set_title('Feature importance for INH')
ax[1].set_ylabel('Feature')
ax[2].set_xlabel('Decision Trees')
ax[2].set_title('Feature importance for RIF')
ax[2].set_ylabel('Feature')
plt.show()

#We take the most important positions for each classification, arranging by weight, referring to the number of trees in which it has appeared

importancias1_RA = classifier1_reduccion.estimators_[0].get_booster().get_score(importance_type= 'weight')
importancias2_RA = classifier1_reduccion.estimators_[1].get_booster().get_score(importance_type= 'weight')
importancias3_RA = classifier1_reduccion.estimators_[2].get_booster().get_score(importance_type= 'weight')

importancia_ordenada1_RA = sorted(importancias1_RA.items(), key=lambda x: x[1], reverse=True)
caracteristicas1 = [x[0] for x in importancia_ordenada1_RA]
print(caracteristicas1)

importancia_ordenada2_RA = sorted(importancias2_RA.items(), key=lambda x: x[1], reverse=True)
caracteristicas2 = [x[0] for x in importancia_ordenada2_RA]
print(caracteristicas2)

importancia_ordenada3_RA = sorted(importancias3_RA.items(), key=lambda x: x[1], reverse=True)
caracteristicas3 = [x[0] for x in importancia_ordenada3_RA]
print(caracteristicas3)

# We make a list of the columns that appear in the decision trees

lista_importancias1_RA = list(sorted(importancias1_RA, key=importancias1_RA.get, reverse=True))
lista_importancias2_RA = list(sorted(importancias2_RA, key=importancias2_RA.get, reverse=True))
lista_importancias3_RA = list(sorted(importancias3_RA, key=importancias3_RA.get, reverse=True))

print(len(lista_importancias1_RA))
print(len(lista_importancias2_RA))
print(len(lista_importancias3_RA))


"""**GENOMIC POSITION BY COLUMN #**"""

# Find feature by column number in original dataset

datasetRS.iloc[:,1].name

datasetRS.columns[74510]

# Find feature by column number in arbitrary reduction dataset (RA)

datasetRS_reduccion.iloc[:,1].name

datasetRS_reduccion.columns[58]

##### Training normal and PCA models ######

def train_and_evaluate_models():
    # ---- Model 1: classifier1_pca ----
    print("Training classifier1_pca...")
    start_time = time.time()
    classifier1_pca.fit(X_train_pca, y_train_pca)
    print("Train score:", classifier1_pca.score(X_train_pca, y_train_pca))
    yhat1_pca = classifier1_pca.predict(X_test_pca)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier1_pca.score(X_test_pca, y_test_pca))
    fig, ax = plt.subplots(ncols=3, figsize=(20, 8))
    for i, drug in enumerate(["EMB", "INH", "RIF"]):
        plot_importance(classifier1_pca.estimators_[i], height=0.5, ax=ax[i], max_num_features=20)
        ax[i].set_xlabel('Decision Trees')
        ax[i].set_ylabel('Feature')
        ax[i].set_title(f'Feature Importance for {drug}')
    plt.show()

    # ---- Model 2: classifier2 ----
    print("Training classifier2...")
    start_time = time.time()
    classifier2.fit(x_trainr, y_trainr)
    print("Train score:", classifier2.score(x_trainr, y_trainr))
    yhat2 = classifier2.predict(x_testr)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier2.score(x_testr, y_testr))
    fig, ax = plt.subplots(ncols=3, figsize=(31, 8))
    for i, drug in enumerate(["EMB", "INH", "RIF"]):
        lightgbm.plot_importance(classifier2.estimators_[i], height=0.5, ax=ax[i], max_num_features=20)
        ax[i].set_xlabel('Decision Trees')
        ax[i].set_ylabel('Feature')
        ax[i].set_title(f'Feature Importance for {drug}')
    plt.show()

    # ---- Model 3: classifier2_pca ----
    print("Training classifier2_pca...")
    start_time = time.time()
    classifier2_pca.fit(X_train_pca, y_train_pca)
    print("Train score:", classifier2_pca.score(X_train_pca, y_train_pca))
    yhat2_pca = classifier2_pca.predict(X_test_pca)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier2_pca.score(X_test_pca, y_test_pca))
    fig, ax = plt.subplots(ncols=3, figsize=(28, 8))
    for i, drug in enumerate(["EMB", "INH", "RIF"]):
        lightgbm.plot_importance(classifier2_pca.estimators_[i], height=0.5, ax=ax[i], max_num_features=20)
        ax[i].set_xlabel('Decision Trees')
        ax[i].set_ylabel('Feature')
        ax[i].set_title(f'Feature Importance for {drug}')
    plt.show()

    # ---- Model 4: classifier3 ----
    print("Training classifier3...")
    start_time = time.time()
    classifier3.fit(x_trainr, y_trainr)
    print("Train score:", classifier3.score(x_trainr, y_trainr))
    yhat3 = classifier3.predict(x_testr)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier3.score(x_testr, y_testr))

    # ---- Model 5: classifier3_pca ----
    print("Training classifier3_pca...")
    start_time = time.time()
    classifier3_pca.fit(X_train_pca, y_train_pca)
    print("Train score:", classifier3_pca.score(X_train_pca, y_train_pca))
    yhat3_pca = classifier3_pca.predict(X_test_pca)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier3_pca.score(X_test_pca, y_test_pca))

    # ---- Model 6: classifier4 ----
    print("Training classifier4...")
    start_time = time.time()
    classifier4.fit(x_trainr, y_trainr)
    print("Train score:", classifier4.score(x_trainr, y_trainr))
    yhat4 = classifier4.predict(x_testr)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier4.score(x_testr, y_testr))

    # ---- Model 7: classifier4_pca ----
    print("Training classifier4_pca...")
    start_time = time.time()
    classifier4_pca.fit(X_train_pca, y_train_pca)
    print("Train score:", classifier4_pca.score(X_train_pca, y_train_pca))
    yhat4_pca = classifier4_pca.predict(X_test_pca)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Test score:", classifier4_pca.score(X_test_pca, y_test_pca))

    return yhat1_pca, yhat2, yhat2_pca, yhat3, yhat3_pca, yhat4, yhat4_pca

yhat1_pca, yhat2, yhat2_pca, yhat3, yhat3_pca, yhat4, yhat4_pca = train_and_evaluate_models()

###### Confusion Matrix #######

auc_y1 = roc_auc_score(y_testr[:,0],yhat1[:,0])
auc_y2 = roc_auc_score(y_testr[:,1],yhat1[:,1])
auc_y3 = roc_auc_score(y_testr[:,2],yhat1[:,2])

print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f" % (auc_y1, auc_y2, auc_y3))

auc_y1_1 = roc_auc_score(y_testr[:,0],yhat2[:,0])
auc_y2_1 = roc_auc_score(y_testr[:,1],yhat2[:,1])
auc_y3_1 = roc_auc_score(y_testr[:,2],yhat2[:,2])

print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f" % (auc_y1_1, auc_y2_1, auc_y3_1))

auc_y1_2 = roc_auc_score(y_testr[:,0],yhat3[:,0])
auc_y2_2 = roc_auc_score(y_testr[:,1],yhat3[:,1])
auc_y3_2 = roc_auc_score(y_testr[:,2],yhat3[:,2])

print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f" % (auc_y1_2, auc_y2_2, auc_y3_2))

auc_y1_3 = roc_auc_score(y_testr[:,0],yhat4[:,0])
auc_y2_3 = roc_auc_score(y_testr[:,1],yhat4[:,1])
auc_y3_3 = roc_auc_score(y_testr[:,2],yhat4[:,2])

print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f" % (auc_y1_3, auc_y2_3, auc_y3_3))

cm1 = skm.multilabel_confusion_matrix(y_testr, yhat1)
print(cm1)
print(skm.classification_report(y_testr,yhat1))

cm1_pca = skm.multilabel_confusion_matrix(y_test_pca, yhat1_pca)
print(cm1_pca)
print(skm.classification_report(y_testr,yhat1_pca))

cm1_reduccion = skm.multilabel_confusion_matrix(y_test_reduccion, yhat1_reduccion)
print(cm1_reduccion)
print(skm.classification_report(y_testr,yhat1_reduccion))

cm2 = skm.multilabel_confusion_matrix(y_testr, yhat2)
print(cm2)
print(skm.classification_report(y_testr,yhat2))

cm2_pca = skm.multilabel_confusion_matrix(y_test_pca, yhat2_pca)
print(cm2_pca)
print(skm.classification_report(y_test_pca,yhat2_pca))

cm3 = skm.multilabel_confusion_matrix(y_testr, yhat3)
print(cm3)
print(skm.classification_report(y_testr,yhat3))

cm3_pca = skm.multilabel_confusion_matrix(y_test_pca, yhat3_pca)
print(cm3_pca)
print(skm.classification_report(y_test_pca,yhat3_pca))

cm4 = skm.multilabel_confusion_matrix(y_testr, yhat4)
print(cm4)
print(skm.classification_report(y_testr,yhat4))

cm4_pca = skm.multilabel_confusion_matrix(y_test_pca, yhat4_pca)
print(cm4_pca)
print(skm.classification_report(y_test_pca,yhat4_pca))

# Confusion Matrix for XGBC

cm_y1_1 = confusion_matrix(y_testr[:,0],yhat1[:,0])
cm_y2_1 = confusion_matrix(y_testr[:,1],yhat1[:,1])
cm_y3_1 = confusion_matrix(y_testr[:,2],yhat1[:,2])

TN1r = cm_y1_1[0,0]
FP1r = cm_y1_1[0,1]
FN1r = cm_y1_1[1,0]
TP1r = cm_y1_1[1,1]

TN2r = cm_y2_1[0,0]
FP2r = cm_y2_1[0,1]
FN2r = cm_y2_1[1,0]
TP2r = cm_y2_1[1,1]

TN3r = cm_y3_1[0,0]
FP3r = cm_y3_1[0,1]
FN3r = cm_y3_1[1,0]
TP3r = cm_y3_1[1,1]

sensitivityrfr = (TP1r/(TP1r+FN1r))
specificityrfr = (TN1r/(FP1r+TN1r))
recallrfr = TP1r/(TP1r+FN1r)
precisionrfr = TP1r/(TP1r+FP1r)
f1scorerfr = ((2*sensitivityrfr*precisionrfr)/(sensitivityrfr+precisionrfr))

print('XGBC Sensitivity for EMB:', sensitivityrfr)
print('XGBC Specificity for EMB:', specificityrfr)
print('XGBC Recall for EMB:' ,recallrfr)
print('XGBC Accuracy for EMB:' ,precisionrfr)
print('XGBC F1-Score for EMB:' ,f1scorerfr)
print("\n")

sensitivitysvmr = (TP2r/(TP2r+FN2r))
specificitysvmr = (TN2r/(FP2r+TN2r))
recallsvmr = TP2r/(TP2r+FN2r)
precisionsvmr = TP2r/(TP2r+FP2r)
f1scoresvmr = ((2*sensitivitysvmr*precisionsvmr)/(sensitivitysvmr+precisionsvmr))

print('XGBC Sensitivity for INH:', sensitivitysvmr)
print('XGBC Specificity for INH:', specificitysvmr)
print('XGBC Recall for INH:' ,recallsvmr)
print('XGBC Accuracy for INH:' ,precisionsvmr)
print('XGBC F1-Score for INH:' ,f1scoresvmr)
print("\n")

sensitivityannr = (TP3r/(TP3r+FN3r))
specificityannr = (TN3r/(FP3r+TN3r))
recallannr = TP3r/(TP3r+FN3r)
precisionannr = TP3r/(TP3r+FP3r)
f1scoreannr = ((2*sensitivityannr*precisionannr)/(sensitivityannr+precisionannr))

print('XGBC Sensitivity for RIF:', sensitivityannr)
print('XGBC Specificity for RIF:', specificityannr)
print('XGBC Recall for RIF:' ,recallannr)
print('XGBC Accuracy for RIF:' ,precisionannr)
print('XGBC F1-Score for RIF:' ,f1scoreannr)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_1,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_1,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_1,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC - Confusion Matrix for RIF")

# Confusion Matrix for XGBC con PCA

cm_y1_1_pca = confusion_matrix(y_test_pca[:,0],yhat1_pca[:,0])
cm_y2_1_pca = confusion_matrix(y_test_pca[:,1],yhat1_pca[:,1])
cm_y3_1_pca = confusion_matrix(y_test_pca[:,2],yhat1_pca[:,2])

TN1r_pca = cm_y1_1_pca[0,0]
FP1r_pca = cm_y1_1_pca[0,1]
FN1r_pca = cm_y1_1_pca[1,0]
TP1r_pca = cm_y1_1_pca[1,1]

TN2r_pca = cm_y2_1_pca[0,0]
FP2r_pca = cm_y2_1_pca[0,1]
FN2r_pca = cm_y2_1_pca[1,0]
TP2r_pca = cm_y2_1_pca[1,1]

TN3r_pca = cm_y3_1_pca[0,0]
FP3r_pca = cm_y3_1_pca[0,1]
FN3r_pca = cm_y3_1_pca[1,0]
TP3r_pca = cm_y3_1_pca[1,1]

sensitivityrfr_pca = TP1r_pca/(TP1r_pca+FN1r_pca)
specificityrfr_pca = TN1r_pca/(FP1r_pca+TN1r_pca)
recallrfr_pca      = TP1r_pca/(TP1r_pca+FN1r_pca)
precisionrfr_pca   = TP1r_pca/(TP1r_pca+FP1r_pca)
f1scorerfr_pca = ((2*sensitivityrfr_pca*precisionrfr_pca)/(sensitivityrfr_pca+precisionrfr_pca))

print('XGBC-PCA Sensitivity for EMB:', sensitivityrfr_pca)
print('XGBC-PCA Specificity for EMB:', specificityrfr_pca)
print('XGBC-PCA Recall for EMB:'     , recallrfr_pca)
print('XGBC-PCA Accuracy for EMB:'  , precisionrfr_pca)
print('XGBC-PCA F1-Score for EMB:'  , f1scorerfr_pca)
print("\n")

sensitivitysvmr_pca = TP2r_pca/(TP2r_pca+FN2r_pca)
specificitysvmr_pca = TN2r_pca/(FP2r_pca+TN2r_pca)
recallsvmr_pca      = TP2r_pca/(TP2r_pca+FN2r_pca)
precisionsvmr_pca   = TP2r_pca/(TP2r_pca+FP2r_pca)
f1scoresvmr_pca = ((2*sensitivitysvmr_pca*precisionsvmr_pca)/(sensitivitysvmr_pca+precisionsvmr_pca))

print('XGBC-PCA Sensitivity for INH:', sensitivitysvmr_pca)
print('XGBC-PCA Specificity for INH:', specificitysvmr_pca)
print('XGBC-PCA Recall for INH:'     ,recallsvmr_pca)
print('XGBC-PCA Precision for INH:'  ,precisionsvmr_pca)
print('XGBC-PCA F1-Score for INH:'  , f1scoresvmr_pca)
print("\n")

sensitivityannr_pca = TP3r_pca/(TP3r_pca+FN3r_pca)
specificityannr_pca = TN3r_pca/(FP3r_pca+TN3r_pca)
recallannr_pca      = TP3r_pca/(TP3r_pca+FN3r_pca)
precisionannr_pca   = TP3r_pca/(TP3r_pca+FP3r_pca)
f1scoreannr_pca = ((2*sensitivityannr_pca*precisionannr_pca)/(sensitivityannr_pca+precisionannr_pca))

print('XGBC-PCA Sensitivity for RIF:', sensitivityannr_pca)
print('XGBC-PCA Specificity for RIF:', specificityannr_pca)
print('XGBC-PCA Recall for RIF:'     , recallannr_pca)
print('XGBC-PCA Precision for RIF:'  , precisionannr_pca)
print('XGBC-PCA F1-Score for RIF:'  , f1scoreannr_pca)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_1_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-PCA - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_1_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-PCA - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_1_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-PCA - Confusion Matrix for RIF")

# Confusion Matrix for XGBC reduccion arbitraria (RA)

cm_y1_1_reduccion = confusion_matrix(y_test_reduccion[:,0],yhat1_reduccion[:,0])
cm_y2_1_reduccion = confusion_matrix(y_test_reduccion[:,1],yhat1_reduccion[:,1])
cm_y3_1_reduccion = confusion_matrix(y_test_reduccion[:,2],yhat1_reduccion[:,2])

TN1r_reduccion = cm_y1_1_reduccion[0,0]
FP1r_reduccion = cm_y1_1_reduccion[0,1]
FN1r_reduccion = cm_y1_1_reduccion[1,0]
TP1r_reduccion = cm_y1_1_reduccion[1,1]

TN2r_reduccion = cm_y2_1_reduccion[0,0]
FP2r_reduccion = cm_y2_1_reduccion[0,1]
FN2r_reduccion = cm_y2_1_reduccion[1,0]
TP2r_reduccion = cm_y2_1_reduccion[1,1]

TN3r_reduccion = cm_y3_1_reduccion[0,0]
FP3r_reduccion = cm_y3_1_reduccion[0,1]
FN3r_reduccion = cm_y3_1_reduccion[1,0]
TP3r_reduccion = cm_y3_1_reduccion[1,1]

sensitivityrfr_reduccion = TP1r_reduccion/(TP1r_reduccion+FN1r_reduccion)
specificityrfr_reduccion = TN1r_reduccion/(FP1r_reduccion+TN1r_reduccion)
recallrfr_reduccion      = TP1r_reduccion/(TP1r_reduccion+FN1r_reduccion)
precisionrfr_reduccion   = TP1r_reduccion/(TP1r_reduccion+FP1r_reduccion)
f1scorerfr_reduccion = ((2*sensitivityrfr_reduccion*precisionrfr_reduccion)/(sensitivityrfr_reduccion+precisionrfr_reduccion))

print('XGBC-RA Sensitivity for EMB:', sensitivityrfr_reduccion)
print('XGBC-RA Specificity for EMB:', specificityrfr_reduccion)
print('XGBC-RA Recall for EMB:'     , recallrfr_reduccion)
print('XGBC-RA Accuracy for EMB:'  , precisionrfr_reduccion)
print('XGBC-RA F1-Score for EMB:'  , f1scorerfr_reduccion)
print("\n")

sensitivitysvmr_reduccion = TP2r_reduccion/(TP2r_reduccion+FN2r_reduccion)
specificitysvmr_reduccion = TN2r_reduccion/(FP2r_reduccion+TN2r_reduccion)
recallsvmr_reduccion      = TP2r_reduccion/(TP2r_reduccion+FN2r_reduccion)
precisionsvmr_reduccion   = TP2r_reduccion/(TP2r_reduccion+FP2r_reduccion)
f1scoresvmr_reduccion = ((2*sensitivitysvmr_reduccion*precisionsvmr_reduccion)/(sensitivitysvmr_reduccion+precisionsvmr_reduccion))

print('XGBC-RA Sensitivity for INH:', sensitivitysvmr_reduccion)
print('XGBC-RA Specificity for INH:', specificitysvmr_reduccion)
print('XGBC-RA Recall for INH:'     ,recallsvmr_reduccion)
print('XGBC-RA Precision for INH:'  ,precisionsvmr_reduccion)
print('XGBC-RA F1-Score for INH:'  , f1scoresvmr_reduccion)
print("\n")

sensitivityannr_reduccion = TP3r_reduccion/(TP3r_reduccion+FN3r_reduccion)
specificityannr_reduccion = TN3r_reduccion/(FP3r_reduccion+TN3r_reduccion)
recallannr_reduccion      = TP3r_reduccion/(TP3r_reduccion+FN3r_reduccion)
precisionannr_reduccion   = TP3r_reduccion/(TP3r_reduccion+FP3r_reduccion)
f1scoreannr_reduccion = ((2*sensitivityannr_reduccion*precisionannr_reduccion)/(sensitivityannr_reduccion+precisionannr_reduccion))

print('XGBC-RA Sensitivity for RIF:', sensitivityannr_reduccion)
print('XGBC-RA Specificity for RIF:', specificityannr_reduccion)
print('XGBC-RA Recall for RIF:'     , recallannr_reduccion)
print('XGBC-RA Precision for RIF:'  , precisionannr_reduccion)
print('XGBC-RA F1-Score for RIF:'  , f1scoreannr_reduccion)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_1_reduccion,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-RA - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_1_reduccion,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-RA - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_1_reduccion,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "XGBC-RA - Confusion Matrix for RIF")

# Confusion Matrix for LGBC

cm_y1_2 = confusion_matrix(y_testr[:,0],yhat2[:,0])
cm_y2_2 = confusion_matrix(y_testr[:,1],yhat2[:,1])
cm_y3_2 = confusion_matrix(y_testr[:,2],yhat2[:,2])

TN1r = cm_y1_2[0,0]
FP1r = cm_y1_2[0,1]
FN1r = cm_y1_2[1,0]
TP1r = cm_y1_2[1,1]

TN2r = cm_y2_2[0,0]
FP2r = cm_y2_2[0,1]
FN2r = cm_y2_2[1,0]
TP2r = cm_y2_2[1,1]

TN3r = cm_y3_2[0,0]
FP3r = cm_y3_2[0,1]
FN3r = cm_y3_2[1,0]
TP3r = cm_y3_2[1,1]

sensitivityrfr = (TP1r/(TP1r+FN1r))
specificityrfr = (TN1r/(FP1r+TN1r))
recallrfr = TP1r/(TP1r+FN1r)
precisionrfr = TP1r/(TP1r+FP1r)
f1scorerfr = ((2*sensitivityrfr*precisionrfr)/(sensitivityrfr+precisionrfr))

print('LGBC Sensitivity for EMB:', sensitivityrfr)
print('LGBC Specificity for EMB:', specificityrfr)
print('LGBC Recall for EMB:' ,recallrfr)
print('LGBC Accuracy for EMB:' ,precisionrfr)
print('LGBC F1-Score for EMB:' ,f1scorerfr)
print("\n")

sensitivitysvmr = (TP2r/(TP2r+FN2r))
specificitysvmr = (TN2r/(FP2r+TN2r))
recallsvmr = TP2r/(TP2r+FN2r)
precisionsvmr = TP2r/(TP2r+FP2r)
f1scoresvmr = ((2*sensitivitysvmr*precisionsvmr)/(sensitivitysvmr+precisionsvmr))

print('LGBC Sensitivity for INH:', sensitivitysvmr)
print('LGBC Specificity for INH:', specificitysvmr)
print('LGBC Recall for INH:' ,recallsvmr)
print('LGBC Accuracy for INH:' ,precisionsvmr)
print('LGBC F1-Score for INH:' ,f1scoresvmr)
print("\n")

sensitivityannr = (TP3r/(TP3r+FN3r))
specificityannr = (TN3r/(FP3r+TN3r))
recallannr = TP3r/(TP3r+FN3r)
precisionannr = TP3r/(TP3r+FP3r)
f1scoreannr = ((2*sensitivityannr*precisionannr)/(sensitivityannr+precisionannr))

print('LGBC Sensitivity for RIF:', sensitivityannr)
print('LGBC Specificity for RIF:', specificityannr)
print('LGBC Recall for RIF:' ,recallannr)
print('LGBC Accuracy for RIF:' ,precisionannr)
print('LGBC F1-Score for RIF:' ,f1scoreannr)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_2,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_2,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_2,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC - Confusion Matrix for RIF")

# Confusion Matrix for LGB con PCA

cm_y1_2_pca = confusion_matrix(y_test_pca[:,0],yhat2_pca[:,0])
cm_y2_2_pca = confusion_matrix(y_test_pca[:,1],yhat2_pca[:,1])
cm_y3_2_pca = confusion_matrix(y_test_pca[:,2],yhat2_pca[:,2])

TN1r_pca = cm_y1_2_pca[0,0]
FP1r_pca = cm_y1_2_pca[0,1]
FN1r_pca = cm_y1_2_pca[1,0]
TP1r_pca = cm_y1_2_pca[1,1]

TN2r_pca = cm_y2_2_pca[0,0]
FP2r_pca = cm_y2_2_pca[0,1]
FN2r_pca = cm_y2_2_pca[1,0]
TP2r_pca = cm_y2_2_pca[1,1]

TN3r_pca = cm_y3_2_pca[0,0]
FP3r_pca = cm_y3_2_pca[0,1]
FN3r_pca = cm_y3_2_pca[1,0]
TP3r_pca = cm_y3_2_pca[1,1]

sensitivityrfr_pca = TP1r_pca/(TP1r_pca+FN1r_pca)
specificityrfr_pca = TN1r_pca/(FP1r_pca+TN1r_pca)
recallrfr_pca      = TP1r_pca/(TP1r_pca+FN1r_pca)
precisionrfr_pca   = TP1r_pca/(TP1r_pca+FP1r_pca)
f1scorerfr_pca = ((2*sensitivityrfr_pca*precisionrfr_pca)/(sensitivityrfr_pca+precisionrfr_pca))

print('LGBC-PCA Sensitivity for EMB:', sensitivityrfr_pca)
print('LGBC-PCA Specificity for EMB:', specificityrfr_pca)
print('LGBC-PCA Recall for EMB:'     , recallrfr_pca)
print('LGBC-PCA Accuracy for EMB:'  , precisionrfr_pca)
print('LGBC-PCA F1-Score for EMB:'  , f1scorerfr_pca)
print("\n")

sensitivitysvmr_pca = TP2r_pca/(TP2r_pca+FN2r_pca)
specificitysvmr_pca = TN2r_pca/(FP2r_pca+TN2r_pca)
recallsvmr_pca      = TP2r_pca/(TP2r_pca+FN2r_pca)
precisionsvmr_pca   = TP2r_pca/(TP2r_pca+FP2r_pca)
f1scoresvmr_pca = ((2*sensitivitysvmr_pca*precisionsvmr_pca)/(sensitivitysvmr_pca+precisionsvmr_pca))

print('LGBC-PCA Sensitivity for INH:', sensitivitysvmr_pca)
print('LGBC-PCA Specificity for INH:', specificitysvmr_pca)
print('LGBC-PCA Recall for INH:'     ,recallsvmr_pca)
print('LGBC-PCA Precision for INH:'  ,precisionsvmr_pca)
print('LGBC-PCA F1-Score for INH:'  , f1scoresvmr_pca)
print("\n")

sensitivityannr_pca = TP3r_pca/(TP3r_pca+FN3r_pca)
specificityannr_pca = TN3r_pca/(FP3r_pca+TN3r_pca)
recallannr_pca      = TP3r_pca/(TP3r_pca+FN3r_pca)
precisionannr_pca   = TP3r_pca/(TP3r_pca+FP3r_pca)
f1scoreannr_pca = ((2*sensitivityannr_pca*precisionannr_pca)/(sensitivityannr_pca+precisionannr_pca))

print('LGBC-PCA Sensitivity for RIF:', sensitivityannr_pca)
print('LGBC-PCA Specificity for RIF:', specificityannr_pca)
print('LGBC-PCA Recall for RIF:'     , recallannr_pca)
print('LGBC-PCA Precision for RIF:'  , precisionannr_pca)
print('LGBC-PCA F1-Score for RIF:'  , f1scoreannr_pca)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_2_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC-PCA - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_2_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC-PCA - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_2_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "LGBC-PCA - Confusion Matrix for RIF")

# Confusion Matrix for GradientBoosting

cm_y1_3 = confusion_matrix(y_testr[:,0],yhat3[:,0])
cm_y2_3 = confusion_matrix(y_testr[:,1],yhat3[:,1])
cm_y3_3 = confusion_matrix(y_testr[:,2],yhat3[:,2])

TN1r = cm_y1_3[0,0]
FP1r = cm_y1_3[0,1]
FN1r = cm_y1_3[1,0]
TP1r = cm_y1_3[1,1]

TN2r = cm_y2_3[0,0]
FP2r = cm_y2_3[0,1]
FN2r = cm_y2_3[1,0]
TP2r = cm_y2_3[1,1]

TN3r = cm_y3_3[0,0]
FP3r = cm_y3_3[0,1]
FN3r = cm_y3_3[1,0]
TP3r = cm_y3_3[1,1]

sensitivityrfr = (TP1r/(TP1r+FN1r))
specificityrfr = (TN1r/(FP1r+TN1r))
recallrfr = TP1r/(TP1r+FN1r)
precisionrfr = TP1r/(TP1r+FP1r)
f1scorerfr = ((2*sensitivityrfr*precisionrfr)/(sensitivityrfr+precisionrfr))

print('GBC Sensitivity for EMB:', sensitivityrfr)
print('GBC Specificity for EMB:', specificityrfr)
print('GBC Recall for EMB:' ,recallrfr)
print('GBC Accuracy for EMB:' ,precisionrfr)
print('GBC F1-Score for EMB:' ,f1scorerfr)
print("\n")

sensitivitysvmr = (TP2r/(TP2r+FN2r))
specificitysvmr = (TN2r/(FP2r+TN2r))
recallsvmr = TP2r/(TP2r+FN2r)
precisionsvmr = TP2r/(TP2r+FP2r)
f1scoresvmr = ((2*sensitivitysvmr*precisionsvmr)/(sensitivitysvmr+precisionsvmr))

print('GBC Sensitivity for INH:', sensitivitysvmr)
print('GBC Specificity for INH:', specificitysvmr)
print('GBC Recall for INH:' ,recallsvmr)
print('GBC Accuracy for INH:' ,precisionsvmr)
print('GBC F1-Score for INH:' ,f1scoresvmr)
print("\n")

sensitivityannr = (TP3r/(TP3r+FN3r))
specificityannr = (TN3r/(FP3r+TN3r))
recallannr = TP3r/(TP3r+FN3r)
precisionannr = TP3r/(TP3r+FP3r)
f1scoreannr = ((2*sensitivityannr*precisionannr)/(sensitivityannr+precisionannr))

print('GBC Sensitivity for RIF:', sensitivityannr)
print('GBC Specificity for RIF:', specificityannr)
print('GBC Recall for RIF:' ,recallannr)
print('GBC Accuracy for RIF:' ,precisionannr)
print('GBC F1-Score for RIF:' ,f1scoreannr)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_3,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_3,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_3,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC - Confusion Matrix for RIF")

# Confusion Matrix for GradientBoosting con PCA

cm_y1_3_pca = confusion_matrix(y_test_pca[:,0],yhat3_pca[:,0])
cm_y2_3_pca = confusion_matrix(y_test_pca[:,1],yhat3_pca[:,1])
cm_y3_3_pca = confusion_matrix(y_test_pca[:,2],yhat3_pca[:,2])

TN1r_pca = cm_y1_3_pca[0,0]
FP1r_pca = cm_y1_3_pca[0,1]
FN1r_pca = cm_y1_3_pca[1,0]
TP1r_pca = cm_y1_3_pca[1,1]

TN2r_pca = cm_y2_3_pca[0,0]
FP2r_pca = cm_y2_3_pca[0,1]
FN2r_pca = cm_y2_3_pca[1,0]
TP2r_pca = cm_y2_3_pca[1,1]

TN3r_pca = cm_y3_3_pca[0,0]
FP3r_pca = cm_y3_3_pca[0,1]
FN3r_pca = cm_y3_3_pca[1,0]
TP3r_pca = cm_y3_3_pca[1,1]

sensitivityrfr_pca = TP1r_pca/(TP1r_pca+FN1r_pca)
specificityrfr_pca = TN1r_pca/(FP1r_pca+TN1r_pca)
recallrfr_pca      = TP1r_pca/(TP1r_pca+FN1r_pca)
precisionrfr_pca   = TP1r_pca/(TP1r_pca+FP1r_pca)
f1scorerfr_pca = ((2*sensitivityrfr_pca*precisionrfr_pca)/(sensitivityrfr_pca+precisionrfr_pca))

print('GBC-PCA Sensitivity for EMB:', sensitivityrfr_pca)
print('GBC-PCA Specificity for EMB:', specificityrfr_pca)
print('GBC-PCA Recall for EMB:'     , recallrfr_pca)
print('GBC-PCA Accuracy for EMB:'  , precisionrfr_pca)
print('GBC-PCA F1-Score for EMB:'  , f1scorerfr_pca)
print("\n")

sensitivitysvmr_pca = TP2r_pca/(TP2r_pca+FN2r_pca)
specificitysvmr_pca = TN2r_pca/(FP2r_pca+TN2r_pca)
recallsvmr_pca      = TP2r_pca/(TP2r_pca+FN2r_pca)
precisionsvmr_pca   = TP2r_pca/(TP2r_pca+FP2r_pca)
f1scoresvmr_pca = ((2*sensitivitysvmr_pca*precisionsvmr_pca)/(sensitivitysvmr_pca+precisionsvmr_pca))

print('GBC-PCA Sensitivity for INH:', sensitivitysvmr_pca)
print('GBC-PCA Specificity for INH:', specificitysvmr_pca)
print('GBC-PCA Recall for INH:'     ,recallsvmr_pca)
print('GBC-PCA Precision for INH:'  ,precisionsvmr_pca)
print('GBC-PCA F1-Score for INH:'  , f1scoresvmr_pca)
print("\n")

sensitivityannr_pca = TP3r_pca/(TP3r_pca+FN3r_pca)
specificityannr_pca = TN3r_pca/(FP3r_pca+TN3r_pca)
recallannr_pca      = TP3r_pca/(TP3r_pca+FN3r_pca)
precisionannr_pca   = TP3r_pca/(TP3r_pca+FP3r_pca)
f1scoreannr_pca = ((2*sensitivityannr_pca*precisionannr_pca)/(sensitivityannr_pca+precisionannr_pca))

print('GBC-PCA Sensitivity for RIF:', sensitivityannr_pca)
print('GBC-PCA Specificity for RIF:', specificityannr_pca)
print('GBC-PCA Recall for RIF:'     , recallannr_pca)
print('GBC-PCA Precision for RIF:'  , precisionannr_pca)
print('GBC-PCA F1-Score for RIF:'  , f1scoreannr_pca)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_3_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC-PCA - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_3_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC-PCA - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_3_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "GBC-PCA - Confusion Matrix for RIF")

# Confusion Matrix for ANN

cm_y1_4 = confusion_matrix(y_testr[:,0],yhat4[:,0])
cm_y2_4 = confusion_matrix(y_testr[:,1],yhat4[:,1])
cm_y3_4 = confusion_matrix(y_testr[:,2],yhat4[:,2])

TN1r = cm_y1_4[0,0]
FP1r = cm_y1_4[0,1]
FN1r = cm_y1_4[1,0]
TP1r = cm_y1_4[1,1]

TN2r = cm_y2_4[0,0]
FP2r = cm_y2_4[0,1]
FN2r = cm_y2_4[1,0]
TP2r = cm_y2_4[1,1]

TN3r = cm_y3_4[0,0]
FP3r = cm_y3_4[0,1]
FN3r = cm_y3_4[1,0]
TP3r = cm_y3_4[1,1]

sensitivityrfr = (TP1r/(TP1r+FN1r))
specificityrfr = (TN1r/(FP1r+TN1r))
recallrfr = TP1r/(TP1r+FN1r)
precisionrfr = TP1r/(TP1r+FP1r)
f1scorerfr = ((2*sensitivityrfr*precisionrfr)/(sensitivityrfr+precisionrfr))

print('ANN Sensitivity for EMB:', sensitivityrfr)
print('ANN Specificity for EMB:', specificityrfr)
print('ANN Recall for EMB:' ,recallrfr)
print('ANN Accuracy for EMB:' ,precisionrfr)
print('ANN F1-Score for EMB:' ,f1scorerfr)
print("\n")

sensitivitysvmr = (TP2r/(TP2r+FN2r))
specificitysvmr = (TN2r/(FP2r+TN2r))
recallsvmr = TP2r/(TP2r+FN2r)
precisionsvmr = TP2r/(TP2r+FP2r)
f1scoresvmr = ((2*sensitivitysvmr*precisionsvmr)/(sensitivitysvmr+precisionsvmr))

print('ANN Sensitivity for INH:', sensitivitysvmr)
print('ANN Specificity for INH:', specificitysvmr)
print('ANN Recall for INH:' ,recallsvmr)
print('ANN Accuracy for INH:' ,precisionsvmr)
print('ANN F1-Score for INH:' ,f1scoresvmr)
print("\n")

sensitivityannr = (TP3r/(TP3r+FN3r))
specificityannr = (TN3r/(FP3r+TN3r))
recallannr = TP3r/(TP3r+FN3r)
precisionannr = TP3r/(TP3r+FP3r)
f1scoreannr = ((2*sensitivityannr*precisionannr)/(sensitivityannr+precisionannr))

print('ANN Sensitivity for RIF:', sensitivityannr)
print('ANN Specificity for RIF:', specificityannr)
print('ANN Recall for RIF:' ,recallannr)
print('ANN Accuracy for RIF:' ,precisionannr)
print('ANN F1-Score for RIF:' ,f1scoreannr)
print("\n")

# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_4,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_4,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_4,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN - Confusion Matrix for RIF")

# Confusion Matrix for ANN con PCA

cm_y1_4_pca = confusion_matrix(y_test_pca[:,0],yhat4_pca[:,0])
cm_y2_4_pca = confusion_matrix(y_test_pca[:,1],yhat4_pca[:,1])
cm_y3_4_pca = confusion_matrix(y_test_pca[:,2],yhat4_pca[:,2])

TN1r_pca = cm_y1_4_pca[0,0]
FP1r_pca = cm_y1_4_pca[0,1]
FN1r_pca = cm_y1_4_pca[1,0]
TP1r_pca = cm_y1_4_pca[1,1]

TN2r_pca = cm_y2_4_pca[0,0]
FP2r_pca = cm_y2_4_pca[0,1]
FN2r_pca = cm_y2_4_pca[1,0]
TP2r_pca = cm_y2_4_pca[1,1]

TN3r_pca = cm_y3_4_pca[0,0]
FP3r_pca = cm_y3_4_pca[0,1]
FN3r_pca = cm_y3_4_pca[1,0]
TP3r_pca = cm_y3_4_pca[1,1]

sensitivityrfr_pca = TP1r_pca/(TP1r_pca+FN1r_pca)
specificityrfr_pca = TN1r_pca/(FP1r_pca+TN1r_pca)
recallrfr_pca      = TP1r_pca/(TP1r_pca+FN1r_pca)
precisionrfr_pca   = TP1r_pca/(TP1r_pca+FP1r_pca)
f1scorerfr_pca = ((2*sensitivityrfr_pca*precisionrfr_pca)/(sensitivityrfr_pca+precisionrfr_pca))

print('ANN-PCA Sensitivity for EMB:', sensitivityrfr_pca)
print('ANN-PCA Specificity for EMB:', specificityrfr_pca)
print('ANN-PCA Recall for EMB:'     , recallrfr_pca)
print('ANN-PCA Accuracy for EMB:'  , precisionrfr_pca)
print('ANN-PCA F1-Score for EMB:'  , f1scorerfr_pca)
print("\n")

sensitivitysvmr_pca = TP2r_pca/(TP2r_pca+FN2r_pca)
specificitysvmr_pca = TN2r_pca/(FP2r_pca+TN2r_pca)
recallsvmr_pca      = TP2r_pca/(TP2r_pca+FN2r_pca)
precisionsvmr_pca   = TP2r_pca/(TP2r_pca+FP2r_pca)
f1scoresvmr_pca = ((2*sensitivitysvmr_pca*precisionsvmr_pca)/(sensitivitysvmr_pca+precisionsvmr_pca))

print('ANN-PCA Sensitivity for INH:', sensitivitysvmr_pca)
print('ANN-PCA Specificity for INH:', specificitysvmr_pca)
print('ANN-PCA Recall for INH:'     ,recallsvmr_pca)
print('ANN-PCA Precision for INH:'  ,precisionsvmr_pca)
print('ANN-PCA F1-Score for INH:'  , f1scoresvmr_pca)
print("\n")

sensitivityannr_pca = TP3r_pca/(TP3r_pca+FN3r_pca)
specificityannr_pca = TN3r_pca/(FP3r_pca+TN3r_pca)
recallannr_pca      = TP3r_pca/(TP3r_pca+FN3r_pca)
precisionannr_pca   = TP3r_pca/(TP3r_pca+FP3r_pca)
f1scoreannr_pca = ((2*sensitivityannr_pca*precisionannr_pca)/(sensitivityannr_pca+precisionannr_pca))

print('ANN-PCA Sensitivity for RIF:', sensitivityannr_pca)
print('ANN-PCA Specificity for RIF:', specificityannr_pca)
print('ANN-PCA Recall for RIF:'     , recallannr_pca)
print('ANN-PCA Precision for RIF:'  , precisionannr_pca)
print('ANN-PCA F1-Score for RIF:'  , f1scoreannr_pca)
print("\n")
# funcion para graficar

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=25,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Predicción\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm_y1_4_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN-PCA - Confusion Matrix for EMB")

plot_confusion_matrix(cm_y2_4_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN-PCA - Confusion Matrix for INH")

plot_confusion_matrix(cm_y3_4_pca,
                      normalize    = False,
                      target_names = ['susceptible', 'resistant'],
                      title        = "ANN-PCA - Confusion Matrix for RIF")
