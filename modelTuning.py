import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

def saveModel(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    return

def loadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def evaluateClassifier(model, test_features, test_labels):
    accuracy = np.mean(cross_val_score(model, test_features, test_labels, cv=3))
    return accuracy

def parameterTuning(modelType, Xtrain, ytrain, Xtest, ytest, X_presence, y_presence):
    if (modelType == 'RF'):
        classifier = RandomForestClassifier()
        base_class = RandomForestClassifier()
        param_test = {
            'n_estimators':[50, 100, 150, 200],
            'max_depth':[100, 125, 150],
            'min_samples_split':[10, 25, 50, 75],
            'min_samples_leaf':[1, 2, 4, 6, 8],
            'max_features':[2, 3, 4, 5]
            }
    elif (modelType == 'GB'):
        classifier = GradientBoostingClassifier()
        base_class = GradientBoostingClassifier()
        param_test = {
            'n_estimators':[25, 50, 100, 250],
            'max_depth': [2, 5, 10, 15],
            'min_samples_split':[25, 150, 500],
            'min_samples_leaf':[50, 75, 125],
            'max_features':[5, 6, 7, 8, 9],
            'subsample':[0.75, 0.9, 0.95]
             }    
    elif (modelType == 'SVC'):
        classifier = SVC()
        base_class = SVC()
        param_test = {
            'kernel': ['rbf', 'sigmoid'],
            'C': [50, 100, 250, 500, 750, 1000],
            'gamma': ['scale', 'auto']
        }
    elif (modelType == 'KNN'):
        classifier = KNeighborsClassifier()
        base_class = KNeighborsClassifier()
        param_test = {
            'n_neighbors': [5, 10, 25, 50, 75, 100],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [2, 5, 7, 10, 20, 30],
            'p': [1, 2, 3, 4]
        }
    elif (modelType == 'LR'):
        classifier = LogisticRegression()
        base_class = LogisticRegression()
        param_test = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'tol': [1e-2, 1e-4, 1e-6],
            'C': [0.25, 0.5, 0.75, 1, 1.5, 2],
            'solver': ['saga'],
            'max_iter': [250, 500, 1000, 2000, 5000]
        }
        
    base_class.fit(Xtrain,ytrain)
    base_accuracy = evaluateClassifier(base_class, X_presence, y_presence)
    ypred = base_class.predict(Xtest)
    print(classification_report(ytest, ypred))
    
    gsearch1 = GridSearchCV(estimator = classifier,param_grid = param_test, scoring='average_precision', verbose=5, n_jobs=8, cv=5)
    gsearch1.fit(Xtrain,ytrain)
    
    print('\nOptimized Parameters of ' + modelType + ':')
    print(gsearch1.best_params_)
    
    random_accuracy = evaluateClassifier(gsearch1.best_estimator_, X_presence, y_presence)
    ypred = gsearch1.best_estimator_.predict(Xtest)
    print(classification_report(ytest, ypred))
    
    print('\nComparison Results of ' + modelType + ':')
    print('Base Accuracy: {:0.2f}%.'.format(100 * base_accuracy))
    print('Optimized Accuracy: {:0.2f}%.'.format(100 * random_accuracy))
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
    
    saveModel(gsearch1.best_estimator_, 'models\\' + modelType + '.mod')

def crossValidation(model, X_presence, y_presence, cv):
    print(cross_val_score(model, X_presence, y_presence, cv=cv))