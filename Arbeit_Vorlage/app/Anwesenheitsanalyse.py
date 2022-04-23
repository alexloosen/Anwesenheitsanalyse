# Drill-Funktionen importieren
import drill as drill
import modelTuning as modelTuning
import prepareData as prepareData
import validation as validation
import createModel as createModel

import requests
import numpy as np
import json
import pandas as pd
import seaborn as sns
import datetime as dt
from math import sqrt

import matplotlib.pyplot as mplt
import plotly.express as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Clustering
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D

pd.options.mode.chained_assignment = None  # default='warn'

#rooms = {'0':'Daniel', '1':'Felix N#2','2':'Calvin','3':'bigDataLab',
#         '4':'FelixAkku', '7':'Lukasbuero','9':'Felix B. #1','10':'Felix N #1'}

#for room in rooms: 
#    print('Processing ' + rooms[room])
#    df = pd.DataFrame
#    df = drill.get_PIR_data(room=room, presence=True)
#    if not df.empty:
#        df.to_json('data\\' + room + '.json')

#df = drill.get_PIR_data(room='H215', presence=True)
#df.to_json('data\H215.json')
#df = drill.get_PIR_data(room='H216', presence=True)
#df.to_json('data\H216.json')
#df = drill.get_PIR_data(room='H217', presence=True)
#df.to_json('data\H217.json')
#df = drill.get_PIR_data('dfs', False)
#df.to_json('data\test.json')

df = pd.read_json('data\\H217.json') 
df_comp = pd.read_json('data\\living_room.json')

#plt.scatter(df, x='timestamp', y='co2_ppm', color='presence')

df_new = prepareData.preProcessDataset(df)
df_comp = prepareData.preProcessDataset(df_comp)

#df_new.head()

plt.scatter(df_new, x='timestamp', y='co2_ppm', color='presence')

# timestamp, presence, temperatur und humidity entfernen
# temp/humid erhoehen Genauigkeit deutlich, da relativ unverlaesslich -> von zu vielen aeusseren Faktoren abhaengig
df_timestamp = df_new['timestamp']
y_presence = df_new['presence']
#, 'temperature_celsius', 'relative_humidity_percent'
X_presence = df_new.drop(['timestamp', 'second', 'presence', 'temperature_celsius', 'relative_humidity_percent',
                         'upload_date', 'measurement_count', 'battery'], axis=1)
X_presence_scaled, scaler = prepareData.normalize_min_max(X_presence)
#X_presence = X_presence.drop(['second_sin', 'second_cos'], axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_presence, y_presence, test_size=0.2, random_state=1, shuffle=False)
Xtrain_scaled, Xtest_scaled, ytrain_scaled, ytest_scaled = train_test_split(X_presence_scaled, y_presence, test_size=0.2, random_state=1, shuffle=False)

# shift des trainings-sets um n minuten in die Vergangenheit
# -> Test ob Model auch in die Zukunft Erwartungen treffen kann
#ytrain = ytrain.shift(-5)
#ytrain = ytrain.replace(np.nan, 0)

# Create Classifiers and save to disk
classifiers = ['RFC', 'SVC', 'GBC', 'KNN', 'LR']

#for x in classifiers:
#    if (x == 'LR'):
#        createModel.createClassifier(x, Xtrain_scaled, ytrain_scaled)
#    else:
#        createModel.createClassifier(x, Xtrain, ytrain)

accuracies = {}
roc_curves = {}
for x in classifiers:
    model = modelTuning.loadModel('models\\' + x + '.mod')
    if (x == 'LR'):
        ypred = model.predict(Xtest_scaled)
        roc_curves[x] = roc_curve(ytest_scaled, ypred)
    else:
        ypred = model.predict(Xtest)
        roc_curves[x] = roc_curve(ytest, ypred)        
    #rf_base_accuracy = modelTuning.evaluateClassifier(model, X_presence, y_presence)
    #accuracies[x] = rf_base_accuracy
#df_acc = pd.DataFrame(data=accuracies, dtype=float, index=['0'])

createModel.createClassifier('RFC', Xtrain, ytrain)

# Verschiedene Feature Vektoren vergleichen
#df1 = pd.read_json('validation\\d15_noShift.json')
#df2 = pd.read_json('validation\\d15.json')
#df3 = pd.read_json('validation\\temp_hum_d15.json')
#frames = [df1, df2, df3]
#result = pd.concat(frames)

mplt.plot(roc_curves['RFC'][0], roc_curves['RFC'][1],'r-',label = 'RF')
mplt.plot(roc_curves['SVC'][0], roc_curves['SVC'][1],'b-',label = 'SV')
mplt.plot(roc_curves['GBC'][0], roc_curves['GBC'][1],'m-',label = 'GB')
mplt.plot(roc_curves['KNN'][0], roc_curves['KNN'][1],'y-',label = 'KN')
mplt.plot(roc_curves['LR'][0], roc_curves['LR'][1],'c-',label = 'LR')
#mplt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
mplt.plot([0,1],[0,1],'k-',label='random')
mplt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
mplt.legend()
mplt.xlabel('False Positive Rate')
mplt.ylabel('True Positive Rate')
mplt.show()

model = modelTuning.loadModel('models\\' + 'RFC' + '.mod')
ypred = model.predict(Xtest)
rf_base_accuracy = modelTuning.evaluateClassifier(model, X_presence, y_presence)
print(rf_base_accuracy)

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
mplt.xlabel('true label')
mplt.ylabel('predicted label')

print(classification_report(ytest, ypred))

df_valid_class = Xtest.copy()
df_valid_class['timestamp'] = df_timestamp
df_valid_class['prediction'] = ypred
df_valid_class['co2_ppm'] = df_new['co2_ppm']

plt.scatter(df_valid_class, x='timestamp', y='co2_ppm', color='prediction')

df_comp

# Trainiertes Modell auf Wohnzimmer anwenden und plotten
df_timestamp = df_comp['timestamp']
df_comp = df_comp.drop(['timestamp', 'second'], axis=1)

model = modelTuning.loadModel('models\RFC.mod')
ypred = model.predict(df_comp)
df_valid_class = df_comp.copy()
df_valid_class['timestamp'] = df_timestamp
df_valid_class['prediction'] = ypred
df_valid_class['co2_ppm'] = df_comp['co2_ppm']

plt.scatter(df_valid_class, x='timestamp', y='co2_ppm', color='prediction')

df_plot = pd.Series(model.feature_importances_, index=Xtrain.columns).sort_values()
df_plot.plot(kind='barh', legend=False, width=0.8, figsize=(7,7), sort_columns=True)
#plt.show()

# plot feature-importance
if (model.feature_importances_.any):
    feature_imp = pd.Series(model.feature_importances_,index=Xtrain.columns).sort_values(ascending=False)
    # Creating a bar plot
    ax = sns.barplot(x=feature_imp, y=feature_imp.index)
    ax.set(xlabel='Importance', ylabel='Feature')

from mpl_toolkits import mplot3d

modelPCA = PCA(n_components=3)
modelPCA.fit(X_presence_scaled)
X_2D = modelPCA.transform(X_presence_scaled)

df_new['PCA1'] = X_2D[:, 0]
df_new['PCA2'] = X_2D[:, 1]
df_new['PCA3'] = X_2D[:, 2]
sns.lmplot(x="PCA1", y="PCA2", hue='presence', data=df_new, fit_reg=False, scatter_kws={'alpha':0.5})

fig = mplt.figure(figsize=[16.4, 14.8])
ax = mplt.axes(projection='3d')
ax.scatter3D(df_new['PCA1'], df_new['PCA2'], df_new['PCA3'], c=df_new['presence']);

df1 = df_new.head(8000)

df_clus = pd.DataFrame()
df_clus['presence'] = df1['presence']
df_clus['second_sin'] = df1['second_sin']
df_clus['second_cos'] = df1['second_cos']
#df_clus['co2_ppm'] = df1['co2_ppm']
df_clus['co2_ppm_delta1'] = df1['co2_ppm_delta1']
df_clus['co2_ppm_delta5'] = df1['co2_ppm_delta5']
df_clus['dayofweek'] = df1['dayOfWeek']

iso = Isomap(n_components=2)
iso.fit(df_clus)
data_projected = iso.transform(df_clus)

plt.scatter(data_projected, x=data_projected[:, 0], y=data_projected[:, 1], color=df_clus['presence'], opacity=0.85,
           labels={
            "x": "",
            "y": "",
        },)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print(accuracy_score(ytest, y_model))

print(classification_report(ytest, y_model))

df_valid_class = Xtest.copy()
df_valid_class['timestamp'] = df_timestamp
df_valid_class['prediction'] = y_model
df_valid_class['co2_ppm'] = df_new['co2_ppm']

plt.scatter(df_valid_class, x='timestamp', y='co2_ppm', color='prediction')

# unsupervised learning: Clustering
modelKM = GaussianMixture(n_components=2, covariance_type='full')
modelKM.fit(Xtrain)
y_ggm = modelKM.predict(Xtest)
print(accuracy_score(ytest, y_model))

print(classification_report(ytest, y_ggm))

y_ggm = np.logical_not(y_ggm).astype(int)

df_valid_ggm = Xtest.copy()
df_valid_ggm['timestamp'] = df_timestamp
df_valid_ggm['prediction'] = y_ggm
df_valid_ggm['co2_ppm'] = df_new['co2_ppm']

plt.scatter(df_valid_ggm, x='timestamp', y='co2_ppm', color='prediction')
#df_new['gaussian_mixture'] = y_ggm
#plt.scatter(df_new, x='PCA1', y="PCA2", color='gaussian_mixture', opacity=0.5)

kmeans = KMeans(n_clusters=2)
kmeans.fit(Xtrain)
y_km = kmeans.predict(Xtest)
#y_km = np.logical_not(y_km).astype(int)
print(accuracy_score(ytest, y_km))

print(classification_report(ytest, y_km))

#df_new['k_means'] = y_km
#plt.scatter(df_new, x='timestamp', y="co2_ppm", color='k_means')

df_valid_km = Xtest.copy()
df_valid_km['timestamp'] = df_timestamp
df_valid_km['prediction'] = y_km
df_valid_km['co2_ppm'] = df_new['co2_ppm']

plt.scatter(df_valid_km, x='timestamp', y='co2_ppm', color='prediction')

#modelTuning.parameterTuning('RF', Xtrain, ytrain, Xtest, ytest, X_presence, y_presence)
#modelTuning.parameterTuning('GB', Xtrain, ytrain, Xtest, ytest, X_presence, y_presence)
#modelTuning.parameterTuning('SVC', Xtrain, ytrain, Xtest, ytest, X_presence, y_presence)
#modelTuning.parameterTuning('KNN', Xtrain, ytrain, Xtest, ytest, X_presence, y_presence)

X_presence, scaler = prepareData.normalize_min_max(X_presence)
#Xtest, _ = normalize_min_max(Xtrain, scaler)

timesteps = 25
X, y = prepareData.reshape_data_for_LSTM(X_presence, y_presence, timesteps)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

model = createModel.createLSTM(Xtrain)
model.summary()

history_lstm = model.fit(Xtrain, ytrain, epochs=200, batch_size=8, verbose=2, validation_data=(Xtest, ytest))

model = createModel.createNN(Xtrain)
#model.summary()

history = model.fit(X_presence, y_presence, epochs=100, validation_split=0.2)

#val_loss, val_acc = model.evaluate(X, y)
#print(val_loss, val_acc)

mplt = validation.plotHistoryAccuracy(history_lstm)
mplt.show()

mplt = validation.plotHistoryLoss(history_lstm)
mplt.show()

mplt = validation.plotHistoryAccuracy(history)
mplt.show()

mplt = validation.plotHistoryLoss(history)
mplt.show()