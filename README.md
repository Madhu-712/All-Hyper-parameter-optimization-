All Techniques Of Hyper Parameter Optimization
GridSearchCV
RandomizedSearchCV
Bayesian Optimization -Automate Hyperparameter Tuning (Hyperopt)
Sequential Model Based Optimization(Tuning a scikit-learn estimator with skopt)
Optuna- Automate Hyperparameter Tuning
Genetic Algorithms (TPOT Classifier)
References
https://github.com/fmfn/BayesianOptimization
https://github.com/hyperopt/hyperopt
https://www.jeremyjordan.me/hyperparameter-tuning/
https://optuna.org/
https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d(By Pier Paolo Ippolito )
https://scikit-optimize.github.io/stable/auto_examples/hyperparameter-optimization.html
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df=pd.read_csv('diabetes.csv')
df.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
import numpy as np
df['Glucose']=np.where(df['Glucose']==0,df['Glucose'].median(),df['Glucose'])
df.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148.0	72	35	0	33.6	0.627	50	1
1	1	85.0	66	29	0	26.6	0.351	31	0
2	8	183.0	64	0	0	23.3	0.672	32	1
3	1	89.0	66	23	94	28.1	0.167	21	0
4	0	137.0	40	35	168	43.1	2.288	33	1
#### Independent And Dependent features
X=df.drop('Outcome',axis=1)
y=df['Outcome']
pd.DataFrame(X,columns=df.columns[:-1])
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
0	6	148.0	72	35	0	33.6	0.627	50
1	1	85.0	66	29	0	26.6	0.351	31
2	8	183.0	64	0	0	23.3	0.672	32
3	1	89.0	66	23	94	28.1	0.167	21
4	0	137.0	40	35	168	43.1	2.288	33
...	...	...	...	...	...	...	...	...
763	10	101.0	76	48	180	32.9	0.171	63
764	2	122.0	70	27	0	36.8	0.340	27
765	5	121.0	72	23	112	26.2	0.245	30
766	1	126.0	60	0	0	30.1	0.349	47
767	1	93.0	70	31	0	30.4	0.315	23
768 rows × 8 columns

#### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=10).fit(X_train,y_train)
prediction=rf_classifier.predict(X_test)
y.value_counts()
0    500
1    268
Name: Outcome, dtype: int64
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))
[[90 17]
 [22 25]]
0.7467532467532467
              precision    recall  f1-score   support

           0       0.80      0.84      0.82       107
           1       0.60      0.53      0.56        47

    accuracy                           0.75       154
   macro avg       0.70      0.69      0.69       154
weighted avg       0.74      0.75      0.74       154

The main parameters used by a Random Forest Classifier are:

criterion = the function used to evaluate the quality of a split.
max_depth = maximum number of levels allowed in each tree.
max_features = maximum number of features considered when splitting a node.
min_samples_leaf = minimum number of samples which can be stored in a tree leaf.
min_samples_split = minimum number of samples necessary in a node to cause node splitting.
n_estimators = number of trees in the ensamble.
### Manual Hyperparameter Tuning
model=RandomForestClassifier(n_estimators=300,criterion='entropy',
                             max_features='sqrt',min_samples_leaf=10,random_state=100).fit(X_train,y_train)
predictions=model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
[[98  9]
 [18 29]]
0.8246753246753247
              precision    recall  f1-score   support

           0       0.84      0.92      0.88       107
           1       0.76      0.62      0.68        47

    accuracy                           0.82       154
   macro avg       0.80      0.77      0.78       154
weighted avg       0.82      0.82      0.82       154

Randomized Search Cv
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)
{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000], 'min_samples_split': [2, 5, 10, 14], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['entropy', 'gini']}
rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
rf_randomcv.fit(X_train,y_train)
Fitting 3 folds for each of 100 candidates, totalling 300 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    7.5s
[Parallel(n_jobs=-1)]: Done 138 tasks      | elapsed:   39.4s
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  1.5min finished
RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=100,
                   n_jobs=-1,
                   param_distributions={'criterion': ['entropy', 'gini'],
                                        'max_depth': [10, 120, 230, 340, 450,
                                                      560, 670, 780, 890,
                                                      1000],
                                        'max_features': ['auto', 'sqrt',
                                                         'log2'],
                                        'min_samples_leaf': [1, 2, 4, 6, 8],
                                        'min_samples_split': [2, 5, 10, 14],
                                        'n_estimators': [200, 400, 600, 800,
                                                         1000, 1200, 1400, 1600,
                                                         1800, 2000]},
                   random_state=100, verbose=2)
rf_randomcv.best_params_
{'n_estimators': 200,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 450,
 'criterion': 'gini'}
rf_randomcv
RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=100,
                   n_jobs=-1,
                   param_distributions={'criterion': ['entropy', 'gini'],
                                        'max_depth': [10, 120, 230, 340, 450,
                                                      560, 670, 780, 890,
                                                      1000],
                                        'max_features': ['auto', 'sqrt',
                                                         'log2'],
                                        'min_samples_leaf': [1, 2, 4, 6, 8],
                                        'min_samples_split': [2, 5, 10, 14],
                                        'n_estimators': [200, 400, 600, 800,
                                                         1000, 1200, 1400, 1600,
                                                         1800, 2000]},
                   random_state=100, verbose=2)
best_random_grid=rf_randomcv.best_estimator_
from sklearn.metrics import accuracy_score
y_pred=best_random_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))
[[93 14]
 [15 32]]
Accuracy Score 0.8116883116883117
Classification report:               precision    recall  f1-score   support

           0       0.86      0.87      0.87       107
           1       0.70      0.68      0.69        47

    accuracy                           0.81       154
   macro avg       0.78      0.78      0.78       154
weighted avg       0.81      0.81      0.81       154

GridSearch CV
rf_randomcv.best_params_
{'n_estimators': 200,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 450,
 'criterion': 'gini'}
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}

print(param_grid)
{'criterion': ['gini'], 'max_depth': [450], 'max_features': ['log2'], 'min_samples_leaf': [1, 3, 5], 'min_samples_split': [3, 4, 5, 6, 7], 'n_estimators': [0, 100, 200, 300, 400]}
#### Fit the grid_search to the data
rf=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(X_train,y_train)
Fitting 10 folds for each of 75 candidates, totalling 750 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    0.3s
[Parallel(n_jobs=-1)]: Done 186 tasks      | elapsed:    9.5s
[Parallel(n_jobs=-1)]: Done 389 tasks      | elapsed:   20.1s
[Parallel(n_jobs=-1)]: Done 672 tasks      | elapsed:   35.0s
[Parallel(n_jobs=-1)]: Done 750 out of 750 | elapsed:   39.8s finished
GridSearchCV(cv=10, estimator=RandomForestClassifier(), n_jobs=-1,
             param_grid={'criterion': ['gini'], 'max_depth': [450],
                         'max_features': ['log2'],
                         'min_samples_leaf': [1, 3, 5],
                         'min_samples_split': [3, 4, 5, 6, 7],
                         'n_estimators': [0, 100, 200, 300, 400]},
             verbose=2)
grid_search.best_estimator_
RandomForestClassifier(max_depth=450, max_features='log2', min_samples_leaf=3,
                       min_samples_split=3)
best_grid=grid_search.best_estimator_
best_grid
RandomForestClassifier(max_depth=450, max_features='log2', min_samples_leaf=3,
                       min_samples_split=3)
y_pred=best_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))
[[94 13]
 [15 32]]
Accuracy Score 0.8181818181818182
Classification report:               precision    recall  f1-score   support

           0       0.86      0.88      0.87       107
           1       0.71      0.68      0.70        47

    accuracy                           0.82       154
   macro avg       0.79      0.78      0.78       154
weighted avg       0.82      0.82      0.82       154

Automated Hyperparameter Tuning
Automated Hyperparameter Tuning can be done by using techniques such as

Bayesian Optimization
Gradient Descent
Evolutionary Algorithms
Bayesian Optimization
Bayesian optimization uses probability to find the minimum of a function. The final aim is to find the input value to a function which can gives us the lowest possible output value.It usually performs better than random,grid and manual search providing better performance in the testing phase and reduced optimization time. In Hyperopt, Bayesian Optimization can be implemented giving 3 three main parameters to the function fmin.

Objective Function = defines the loss function to minimize.
Domain Space = defines the range of input values to test (in Bayesian Optimization this space creates a probability distribution for each of the used Hyperparameters).
Optimization Algorithm = defines the search algorithm to use to select the best input values to use in each new iteration.
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }
space
{'criterion': <hyperopt.pyll.base.Apply at 0x2476401ccc8>,
 'max_depth': <hyperopt.pyll.base.Apply at 0x2476401c588>,
 'max_features': <hyperopt.pyll.base.Apply at 0x2476401ac08>,
 'min_samples_leaf': <hyperopt.pyll.base.Apply at 0x2476401a0c8>,
 'min_samples_split': <hyperopt.pyll.base.Apply at 0x2476401aa88>,
 'n_estimators': <hyperopt.pyll.base.Apply at 0x2476401a1c8>}
def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }
from sklearn.model_selection import cross_val_score
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)
best
100%|███████████████████████████████████████████████| 80/80 [05:51<00:00,  4.39s/trial, best loss: -0.7687591630014661]
{'criterion': 1,
 'max_depth': 1110.0,
 'max_features': 1,
 'min_samples_leaf': 0.015761897600901124,
 'min_samples_split': 0.12204527235107072,
 'n_estimators': 3}
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}


print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])
gini
sqrt
750
best['min_samples_leaf']
0.015761897600901124
trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(X_train,y_train)
predictionforest = trainedforest.predict(X_test)
print(confusion_matrix(y_test,predictionforest))
print(accuracy_score(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc5 = accuracy_score(y_test,predictionforest)
[[98  9]
 [22 25]]
0.7987012987012987
              precision    recall  f1-score   support

           0       0.82      0.92      0.86       107
           1       0.74      0.53      0.62        47

    accuracy                           0.80       154
   macro avg       0.78      0.72      0.74       154
weighted avg       0.79      0.80      0.79       154

Genetic Algorithms
Genetic Algorithms tries to apply natural selection mechanisms to Machine Learning contexts.

Let's immagine we create a population of N Machine Learning models with some predifined Hyperparameters. We can then calculate the accuracy of each model and decide to keep just half of the models (the ones that performs best). We can now generate some offsprings having similar Hyperparameters to the ones of the best models so that go get again a population of N models. At this point we can again caltulate the accuracy of each model and repeate the cycle for a defined number of generations. In this way, just the best models will survive at the end of the process.

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(param)
{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000], 'min_samples_split': [2, 5, 10, 14], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['entropy', 'gini']}
from tpot import TPOTClassifier


tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train)
HBox(children=(FloatProgress(value=0.0, description='Optimization Progress', max=84.0, style=ProgressStyle(des…
Generation 1 - Current best internal CV score: 0.7622442916560563
Generation 2 - Current best internal CV score: 0.7622442916560563
Generation 3 - Current best internal CV score: 0.7622442916560563
Exception ignored in: <function WeakSet.__init__.<locals>._remove at 0x00000247469D40D8>
Traceback (most recent call last):
  File "c:\users\krish naik\anaconda3\envs\myenv1\lib\_weakrefset.py", line 38, in _remove
    def _remove(item, selfref=ref(self)):
stopit.utils.TimeoutException
Generation 4 - Current best internal CV score: 0.7622442916560563
Generation 5 - Current best internal CV score: 0.7622442916560563
Best pipeline: RandomForestClassifier(input_matrix, criterion=gini, max_depth=120, max_features=log2, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
TPOTClassifier(config_dict={'sklearn.ensemble.RandomForestClassifier': {'criterion': ['entropy',
                                                                                      'gini'],
                                                                        'max_depth': [10,
                                                                                      120,
                                                                                      230,
                                                                                      340,
                                                                                      450,
                                                                                      560,
                                                                                      670,
                                                                                      780,
                                                                                      890,
                                                                                      1000],
                                                                        'max_features': ['auto',
                                                                                         'sqrt',
                                                                                         'log2'],
                                                                        'min_samples_leaf': [1,
                                                                                             2,
                                                                                             4,
                                                                                             6,
                                                                                             8],
                                                                        'min_samples_split': [2,
                                                                                              5,
                                                                                              10,
                                                                                              14],
                                                                        'n_estimators': [200,
                                                                                         400,
                                                                                         600,
                                                                                         800,
                                                                                         1000,
                                                                                         1200,
                                                                                         1400,
                                                                                         1600,
                                                                                         1800,
                                                                                         2000]}},
               cv=4, early_stop=12, generations=5,
               log_file=<ipykernel.iostream.OutStream object at 0x0000024747072448>,
               offspring_size=12, population_size=24, scoring='accuracy',
               verbosity=2)
accuracy = tpot_classifier.score(X_test, y_test)
print(accuracy)
0.8181818181818182
Optimize hyperparameters of the model using Optuna
The hyperparameters of the above algorithm are n_estimators and max_depth for which we can try different values to see if the model accuracy can be improved. The objective function is modified to accept a trial object. This trial has several methods for sampling hyperparameters. We create a study to run the hyperparameter optimization and finally read the best hyperparameters.

import optuna
import sklearn.svm
def objective(trial):

    classifier = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])
    
    if classifier == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        max_depth = int(trial.suggest_float('max_depth', 10, 100, log=True))

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
    else:
        c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
        
        clf = sklearn.svm.SVC(C=c, gamma='auto')

    return sklearn.model_selection.cross_val_score(
        clf,X_train,y_train, n_jobs=-1, cv=3).mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
[I 2020-07-22 17:29:25,669] Finished trial#0 with value: 0.7459190180137095 with parameters: {'classifier': 'RandomForest', 'n_estimators': 560, 'max_depth': 45.428344785672444}. Best is trial#0 with value: 0.7459190180137095.
[I 2020-07-22 17:29:25,762] Finished trial#1 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 382014186.2292944}. Best is trial#0 with value: 0.7459190180137095.
[I 2020-07-22 17:29:27,152] Finished trial#2 with value: 0.7524469950581859 with parameters: {'classifier': 'RandomForest', 'n_estimators': 990, 'max_depth': 42.25927339838532}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:29,066] Finished trial#3 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1420, 'max_depth': 27.841871988819953}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:31,190] Finished trial#4 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1560, 'max_depth': 33.82604471162079}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:32,336] Finished trial#5 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 810, 'max_depth': 38.563300841598476}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:32,430] Finished trial#6 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 116.01096494017403}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:32,517] Finished trial#7 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 2.298825620064036e-06}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:33,893] Finished trial#8 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 970, 'max_depth': 64.15168265175629}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:34,722] Finished trial#9 with value: 0.744293001753547 with parameters: {'classifier': 'RandomForest', 'n_estimators': 570, 'max_depth': 12.111594105838392}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:37,410] Finished trial#10 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1950, 'max_depth': 96.32160637795971}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:39,428] Finished trial#11 with value: 0.7524390243902439 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1390, 'max_depth': 16.866832427669866}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:41,263] Finished trial#12 with value: 0.7524390243902439 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1330, 'max_depth': 17.353138766268824}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:43,778] Finished trial#13 with value: 0.75242308305436 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1830, 'max_depth': 18.976699590496274}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:44,157] Finished trial#14 with value: 0.7475370636059302 with parameters: {'classifier': 'RandomForest', 'n_estimators': 210, 'max_depth': 10.277581734452633}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:45,900] Finished trial#15 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1140, 'max_depth': 22.11789321936667}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:48,257] Finished trial#16 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1640, 'max_depth': 61.24717197044995}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:48,343] Finished trial#17 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 1.0373068843430439e-10}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:50,028] Finished trial#18 with value: 0.7475609756097561 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1170, 'max_depth': 13.58624517625075}. Best is trial#2 with value: 0.7524469950581859.
[I 2020-07-22 17:29:51,169] Finished trial#19 with value: 0.7556830862426271 with parameters: {'classifier': 'RandomForest', 'n_estimators': 810, 'max_depth': 26.06992262577861}. Best is trial#19 with value: 0.7556830862426271.
[I 2020-07-22 17:29:51,264] Finished trial#20 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 593045673.4121734}. Best is trial#19 with value: 0.7556830862426271.
[I 2020-07-22 17:29:52,470] Finished trial#21 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 800, 'max_depth': 26.193898724637403}. Best is trial#19 with value: 0.7556830862426271.
[I 2020-07-22 17:29:53,978] Finished trial#22 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1010, 'max_depth': 43.24161116653229}. Best is trial#19 with value: 0.7556830862426271.
[I 2020-07-22 17:29:55,053] Finished trial#23 with value: 0.7573091025027897 with parameters: {'classifier': 'RandomForest', 'n_estimators': 750, 'max_depth': 16.51585932349767}. Best is trial#23 with value: 0.7573091025027897.
[I 2020-07-22 17:29:55,854] Finished trial#24 with value: 0.744293001753547 with parameters: {'classifier': 'RandomForest', 'n_estimators': 480, 'max_depth': 23.534083036669067}. Best is trial#23 with value: 0.7573091025027897.
[I 2020-07-22 17:29:57,030] Finished trial#25 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 770, 'max_depth': 54.286060191398654}. Best is trial#23 with value: 0.7573091025027897.
[I 2020-07-22 17:29:57,602] Finished trial#26 with value: 0.7589430894308943 with parameters: {'classifier': 'RandomForest', 'n_estimators': 330, 'max_depth': 30.02830174010586}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:29:58,082] Finished trial#27 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 270, 'max_depth': 31.618595560356702}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:29:58,703] Finished trial#28 with value: 0.7508130081300813 with parameters: {'classifier': 'RandomForest', 'n_estimators': 370, 'max_depth': 13.897987597367557}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:29:59,728] Finished trial#29 with value: 0.7524390243902439 with parameters: {'classifier': 'RandomForest', 'n_estimators': 710, 'max_depth': 20.944711220128674}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:00,368] Finished trial#30 with value: 0.7507890961262554 with parameters: {'classifier': 'RandomForest', 'n_estimators': 420, 'max_depth': 30.469391836696254}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:01,825] Finished trial#31 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 960, 'max_depth': 38.31448830764323}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:02,848] Finished trial#32 with value: 0.7540730113183485 with parameters: {'classifier': 'RandomForest', 'n_estimators': 650, 'max_depth': 50.49579891136223}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:03,985] Finished trial#33 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 680, 'max_depth': 71.81402577531695}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:04,513] Finished trial#34 with value: 0.7475450342738722 with parameters: {'classifier': 'RandomForest', 'n_estimators': 310, 'max_depth': 92.43030726955858}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:05,437] Finished trial#35 with value: 0.7508050374621393 with parameters: {'classifier': 'RandomForest', 'n_estimators': 590, 'max_depth': 50.443426440902826}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:06,207] Finished trial#36 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 510, 'max_depth': 27.075291591508652}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:06,306] Finished trial#37 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 1.2993364331873203}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:07,572] Finished trial#38 with value: 0.7426669854933844 with parameters: {'classifier': 'RandomForest', 'n_estimators': 850, 'max_depth': 15.598118080606088}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:08,575] Finished trial#39 with value: 0.744285031085605 with parameters: {'classifier': 'RandomForest', 'n_estimators': 640, 'max_depth': 36.70667631325387}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:09,951] Finished trial#40 with value: 0.7540570699824646 with parameters: {'classifier': 'RandomForest', 'n_estimators': 890, 'max_depth': 24.278885906819983}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:11,302] Finished trial#41 with value: 0.7540730113183485 with parameters: {'classifier': 'RandomForest', 'n_estimators': 880, 'max_depth': 24.639493598039277}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:12,378] Finished trial#42 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 710, 'max_depth': 30.152677615795024}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:13,983] Finished trial#43 with value: 0.7491710505340348 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1090, 'max_depth': 21.299836366433247}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:15,310] Finished trial#44 with value: 0.7507970667941973 with parameters: {'classifier': 'RandomForest', 'n_estimators': 900, 'max_depth': 33.72373163980857}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:16,898] Finished trial#45 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1030, 'max_depth': 26.744319304545098}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:16,984] Finished trial#46 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 1.765228116724797e-06}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:18,743] Finished trial#47 with value: 0.7556830862426271 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1240, 'max_depth': 19.536814334499145}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:20,625] Finished trial#48 with value: 0.7475450342738722 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1270, 'max_depth': 18.70415248965657}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:22,754] Finished trial#49 with value: 0.7508050374621393 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1500, 'max_depth': 15.088392240941921}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:23,593] Finished trial#50 with value: 0.7442690897497211 with parameters: {'classifier': 'RandomForest', 'n_estimators': 570, 'max_depth': 11.470531223921249}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:25,426] Finished trial#51 with value: 0.7491869918699187 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1210, 'max_depth': 19.53214190067025}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:26,628] Finished trial#52 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 770, 'max_depth': 24.177542200140888}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:27,967] Finished trial#53 with value: 0.7491869918699187 with parameters: {'classifier': 'RandomForest', 'n_estimators': 930, 'max_depth': 17.205429365298503}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:29,483] Finished trial#54 with value: 0.7459110473457676 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1070, 'max_depth': 28.22494087303447}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:30,495] Finished trial#55 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 640, 'max_depth': 20.18043999110511}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:31,708] Finished trial#56 with value: 0.7491710505340348 with parameters: {'classifier': 'RandomForest', 'n_estimators': 840, 'max_depth': 22.593013535367504}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:32,441] Finished trial#57 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 460, 'max_depth': 25.3463407169904}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:32,532] Finished trial#58 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 30008.546064639668}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:34,479] Finished trial#59 with value: 0.7475689462776981 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1350, 'max_depth': 35.18893159953987}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:36,392] Finished trial#60 with value: 0.7459190180137095 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1250, 'max_depth': 17.52810759361537}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:37,659] Finished trial#61 with value: 0.7459110473457676 with parameters: {'classifier': 'RandomForest', 'n_estimators': 850, 'max_depth': 23.693849001413888}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:39,019] Finished trial#62 with value: 0.744285031085605 with parameters: {'classifier': 'RandomForest', 'n_estimators': 890, 'max_depth': 28.96587659841046}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:40,141] Finished trial#63 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 760, 'max_depth': 33.64843819822842}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:41,520] Finished trial#64 with value: 0.7491630798660928 with parameters: {'classifier': 'RandomForest', 'n_estimators': 970, 'max_depth': 40.77312709628961}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:43,161] Finished trial#65 with value: 0.7491710505340348 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1140, 'max_depth': 24.79718320377235}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:44,686] Finished trial#66 with value: 0.7508130081300813 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1030, 'max_depth': 21.78892585293122}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:45,809] Finished trial#67 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 730, 'max_depth': 15.981758086555269}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:46,755] Finished trial#68 with value: 0.744285031085605 with parameters: {'classifier': 'RandomForest', 'n_estimators': 610, 'max_depth': 19.028172963488487}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:47,595] Finished trial#69 with value: 0.75242308305436 with parameters: {'classifier': 'RandomForest', 'n_estimators': 520, 'max_depth': 13.767850596533478}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:48,866] Finished trial#70 with value: 0.7573091025027897 with parameters: {'classifier': 'RandomForest', 'n_estimators': 820, 'max_depth': 74.47439206198088}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:49,222] Finished trial#71 with value: 0.75242308305436 with parameters: {'classifier': 'RandomForest', 'n_estimators': 210, 'max_depth': 72.17451911098534}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:50,541] Finished trial#72 with value: 0.7394069823051171 with parameters: {'classifier': 'RandomForest', 'n_estimators': 800, 'max_depth': 99.75099049302369}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:51,537] Finished trial#73 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 670, 'max_depth': 80.40293451897855}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:52,840] Finished trial#74 with value: 0.7540650406504065 with parameters: {'classifier': 'RandomForest', 'n_estimators': 880, 'max_depth': 47.21541760312192}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:53,422] Finished trial#75 with value: 0.7508050374621393 with parameters: {'classifier': 'RandomForest', 'n_estimators': 370, 'max_depth': 55.41043514073563}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:54,876] Finished trial#76 with value: 0.7556830862426271 with parameters: {'classifier': 'RandomForest', 'n_estimators': 990, 'max_depth': 45.56806623020355}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:56,228] Finished trial#77 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 970, 'max_depth': 61.005802364636914}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:56,316] Finished trial#78 with value: 0.640068547744301 with parameters: {'classifier': 'SVC', 'svc_c': 7.647818733220187e-06}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:57,904] Finished trial#79 with value: 0.7524310537223019 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1140, 'max_depth': 87.23026678309658}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:30:59,443] Finished trial#80 with value: 0.7508050374621393 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1070, 'max_depth': 43.57895946385289}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:00,709] Finished trial#81 with value: 0.7491710505340348 with parameters: {'classifier': 'RandomForest', 'n_estimators': 830, 'max_depth': 51.44475515363362}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:02,093] Finished trial#82 with value: 0.7459190180137095 with parameters: {'classifier': 'RandomForest', 'n_estimators': 930, 'max_depth': 47.043165242503896}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:03,127] Finished trial#83 with value: 0.7524390243902439 with parameters: {'classifier': 'RandomForest', 'n_estimators': 710, 'max_depth': 47.81405467708041}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:04,534] Finished trial#84 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1020, 'max_depth': 40.07976389933971}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:05,616] Finished trial#85 with value: 0.7475450342738722 with parameters: {'classifier': 'RandomForest', 'n_estimators': 780, 'max_depth': 32.00132869479938}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:06,872] Finished trial#86 with value: 0.7589271480950104 with parameters: {'classifier': 'RandomForest', 'n_estimators': 880, 'max_depth': 57.62379085133717}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:08,201] Finished trial#87 with value: 0.7475530049418141 with parameters: {'classifier': 'RandomForest', 'n_estimators': 930, 'max_depth': 69.51334259544966}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:10,150] Finished trial#88 with value: 0.7556830862426271 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1420, 'max_depth': 58.21916890693221}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:12,336] Finished trial#89 with value: 0.7540650406504065 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1650, 'max_depth': 58.6770907625422}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:14,326] Finished trial#90 with value: 0.7540570699824646 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1420, 'max_depth': 67.41669489829536}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:16,495] Finished trial#91 with value: 0.7459269886816515 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1510, 'max_depth': 55.40995368855331}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:18,328] Finished trial#92 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1290, 'max_depth': 63.675677219798835}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:20,247] Finished trial#93 with value: 0.744293001753547 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1450, 'max_depth': 57.787937579774855}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:22,575] Finished trial#94 with value: 0.7475450342738722 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1640, 'max_depth': 77.339561935967}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:24,248] Finished trial#95 with value: 0.7491790212019768 with parameters: {'classifier': 'RandomForest', 'n_estimators': 1210, 'max_depth': 51.30177472499095}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:25,461] Finished trial#96 with value: 0.7508050374621393 with parameters: {'classifier': 'RandomForest', 'n_estimators': 860, 'max_depth': 64.87873839898693}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:26,479] Finished trial#97 with value: 0.7491551091981509 with parameters: {'classifier': 'RandomForest', 'n_estimators': 740, 'max_depth': 26.263414444248404}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:27,619] Finished trial#98 with value: 0.7540730113183485 with parameters: {'classifier': 'RandomForest', 'n_estimators': 810, 'max_depth': 36.90991074547584}. Best is trial#26 with value: 0.7589430894308943.
[I 2020-07-22 17:31:28,571] Finished trial#99 with value: 0.7491710505340348 with parameters: {'classifier': 'RandomForest', 'n_estimators': 670, 'max_depth': 37.0342538028874}. Best is trial#26 with value: 0.7589430894308943.
Accuracy: 0.7589430894308943
Best hyperparameters: {'classifier': 'RandomForest', 'n_estimators': 330, 'max_depth': 30.02830174010586}
trial
FrozenTrial(number=26, value=0.7589430894308943, datetime_start=datetime.datetime(2020, 7, 22, 17, 29, 57, 32020), datetime_complete=datetime.datetime(2020, 7, 22, 17, 29, 57, 602495), params={'classifier': 'RandomForest', 'n_estimators': 330, 'max_depth': 30.02830174010586}, distributions={'classifier': CategoricalDistribution(choices=('RandomForest', 'SVC')), 'n_estimators': IntUniformDistribution(high=2000, low=200, step=10), 'max_depth': LogUniformDistribution(high=100, low=10)}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=26, state=TrialState.COMPLETE)
study.best_params
{'classifier': 'RandomForest',
 'n_estimators': 330,
 'max_depth': 30.02830174010586}
rf=RandomForestClassifier(n_estimators=330,max_depth=30)
rf.fit(X_train,y_train)
RandomForestClassifier(max_depth=30, n_estimators=330)
y_pred=rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
[[94 13]
 [14 33]]
0.8246753246753247
              precision    recall  f1-score   support

           0       0.87      0.88      0.87       107
           1       0.72      0.70      0.71        47

    accuracy                           0.82       154
   macro avg       0.79      0.79      0.79       154
weighted avg       0.82      0.82      0.82       154

 
