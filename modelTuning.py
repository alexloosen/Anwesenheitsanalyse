from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV

def evaluateClassifier(model, test_features, test_labels):
    ypred = model.predict(test_features)
    accuracy = accuracy_score(test_labels, ypred)
    return accuracy

def parameterTuning(modelType, Xtrain, ytrain, Xtest, ytest):
    if (modelType == 'RF'):
        base_class = RandomForestClassifier()
        param_test = {
            'n_estimators':[25, 50, 100, 250, 500, 1000],
            'max_depth':[5,15,50],
            'min_samples_split':[10,200,800],
            'min_samples_leaf':[5,10,40],
            'max_features':[1,2,3]
            }
    elif (modelType == 'GB'):
        base_class = GradientBoostClassifier()
        param_test = {
            'n_estimators':[50, 100, 250],
            'max_depth': [2,5,10],
            'min_samples_split':[100, 500, 1200],
            'min_samples_leaf':[10,30,50],
            'max_features':[2,3],
            'subsample':[0.6,0.75,0.9]
             }    
    elif (modelType == 'SVC'):
        base_class = SVC()
        param_test = {
            'kernel': ['rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100, 500],
            'gamma': ['scale', 'auto']
        }
    elif (modelType == 'KNN'):
        base_class = KNeighborsClassifier()
        param_test = {
            'n_neighbors': [2,5,10,25,50,100,200],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [15,30,60,90,150],
            'p': [1,2,3,4,5]
        }
        
    base_class.fit(Xtrain,ytrain)
    base_accuracy = evaluateClassifier(base_class, Xtest, ytest)
    
    gsearch1 = GridSearchCV(estimator = base_class,param_grid = param_test, verbose=5, n_jobs=-1, cv=3)
    gsearch1.fit(Xtrain,ytrain)
    
    print('\nOptimized Parameters:')
    print(gsearch1.best_params_)
    
    base_class.fit(Xtrain,ytrain)
    base_accuracy = evaluateClassifier(base_class, Xtest, ytest)
    random_accuracy = evaluateClassifier(gsearch1.best_estimator_, Xtest, ytest)
    
    print('\nComparison Results:')
    print('Base Accuracy: {:0.2f}%.'.format(100 * base_accuracy))
    print('Optimized Accuracy: {:0.2f}%.'.format(100 * random_accuracy))
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

def crossValidation(model, X_presence, y_presence, cv):
    print(cross_val_score(model, X_presence, y_presence, cv=cv))