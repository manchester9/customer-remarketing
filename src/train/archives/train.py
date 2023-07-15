# base train code 
import pandas as pd
import numpy as np
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import category_encoders as ce

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from tqdm import tqdm

def traget_variable(data):
    # save target variable before transformation
    y = data.y
    # Create target encoder object and transoform two value
    target_encode = ce.target_encoder.TargetEncoder(cols=['marital', 'education']).fit(data, y)
    numeric_dataset = target_encode.transform(data)
    # drop target variable
    numeric_dataset = numeric_dataset.drop('y', axis=1, inplace=True)
    return numeric_dataset
    '''Check numerical data set'''
    display(numeric_dataset.head(3), numeric_dataset.shape, y.shape)
    display('We observe 41175 rows and 44 numerical features after transformation. Target variable shape is (41175, 0 ) as expected')

def split_dataset(numeric_dataset)
    ''' Split data on train and test'''
    # set global random state
    random_state = 11
    # split data
    X_train, X_test, y_train, y_test = train_test_split(numeric_dataset, y, test_size=0.2, random_state=random_state)
    # collect excess data
    gc.collect()


# cross validation and grid search
def build_pipeline():
    '''Build pipline of classifiers'''
    # set all CPU
    n_jobs = -1
    # LogisticRegression
    pipe_lr = Pipeline([('lr', LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=500))])
    # RandomForestClassifier
    pipe_rf = Pipeline([('rf', RandomForestClassifier(random_state=random_state, oob_score=True, n_jobs=n_jobs))])
    # KNeighborsClassifier
    pipe_knn = Pipeline([('knn', KNeighborsClassifier(n_jobs=n_jobs))])
    # DecisionTreeClassifier
    pipe_dt = Pipeline([('dt', DecisionTreeClassifier(random_state=random_state, max_features='auto'))])
    # BaggingClassifier
    # note we use SGDClassifier as classier inside BaggingClassifier
    pipe_bag = Pipeline([('bag',BaggingClassifier(base_estimator=SGDClassifier(random_state=random_state, n_jobs=n_jobs, max_iter=1500),\
                                                random_state=random_state,oob_score=True,n_jobs=n_jobs))])
    # SGDClassifier
    pipe_sgd = Pipeline([('sgd', SGDClassifier(random_state=random_state, n_jobs=n_jobs, max_iter=1500))])


    '''Set parameters for Grid Search '''
    # set number 
    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=random_state)
    # set for LogisticRegression
    grid_params_lr = [{
                    'lr__penalty': ['l2'],
                    'lr__C': [0.3, 0.6, 0.7],
                    'lr__solver': ['sag']
                    }]
    # set for RandomForestClassifier
    grid_params_rf = [{
                    'rf__criterion': ['entropy'],
                    'rf__min_samples_leaf': [80, 100],
                    'rf__max_depth': [25, 27],
                    'rf__min_samples_split': [3, 5],
                    'rf__n_estimators' : [60, 70]
                    }]
    # set for KNeighborsClassifier
    grid_params_knn = [{'knn__n_neighbors': [16,17,18]}]

    # set for DecisionTreeClassifier
    grid_params_dt = [{
                    'dt__max_depth': [8, 10],
                    'dt__min_samples_leaf': [1, 3, 5, 7]
                    }]
    # set for BaggingClassifier
    grid_params_bag = [{'bag__n_estimators': [10, 15, 20]}]

    # set for SGDClassifier
    grid_params_sgd = [{
                        'sgd__loss': ['log', 'huber'],
                        'sgd__learning_rate': ['adaptive'],
                        'sgd__eta0': [0.001, 0.01, 0.1],
                        'sgd__penalty': ['l1', 'l2', 'elasticnet'], 
                        'sgd__alpha':[0.1, 1, 5, 10]
                        }]
    
    '''Grid search objects'''
    # for LogisticRegression
    gs_lr = GridSearchCV(pipe_lr, param_grid=grid_params_lr,
                        scoring='accuracy', cv=cv) 
    # for RandomForestClassifier
    gs_rf = GridSearchCV(pipe_rf, param_grid=grid_params_rf,
                        scoring='accuracy', cv=cv)
    # for KNeighborsClassifier
    gs_knn = GridSearchCV(pipe_knn, param_grid=grid_params_knn,
                        scoring='accuracy', cv=cv)
    # for DecisionTreeClassifier
    gs_dt = GridSearchCV(pipe_dt, param_grid=grid_params_dt,
                        scoring='accuracy', cv=cv)
    # for BaggingClassifier
    gs_bag = GridSearchCV(pipe_bag, param_grid=grid_params_bag,
                        scoring='accuracy', cv=cv)
    # for SGDClassifier
    gs_sgd = GridSearchCV(pipe_sgd, param_grid=grid_params_sgd,
                        scoring='accuracy', cv=cv)
    
def iterate_models():
    # models that we iterate over
    look_for = [gs_lr, gs_rf, gs_knn, gs_dt, gs_bag, gs_sgd]
    # dict for later use 
    model_dict = {0:'Logistic_reg', 1:'RandomForest', 2:'Knn', 3:'DesionTree', 4:'Bagging with SGDClassifier', 5:'SGD Class'}

    ''' Function to iterate over models and obtain results'''
    # set empty dicts and list
    result_acc = {}
    result_auc = {}
    models = []

    for index, model in enumerate(look_for):
            start = time.time()
            print()
            print('+++++++ Start New Model ++++++++++++++++++++++')
            print('Estimator is {}'.format(model_dict[index]))
            model.fit(X_train, y_train)
            print('---------------------------------------------')
            print('best params {}'.format(model.best_params_))
            print('best score is {}'.format(model.best_score_))
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            print('---------------------------------------------')
            print('ROC_AUC is {} and accuracy rate is {}'.format(auc, model.score(X_test, y_test)))
            end = time.time()
            print('It lasted for {} sec'.format(round(end - start, 3)))
            print('++++++++ End Model +++++++++++++++++++++++++++')
            print()
            print()
            models.append(model.best_estimator_)
            result_acc[index] = model.best_score_
            result_auc[index] = auc

def model_performance():
    """ Model performance during Grid Search """
    performance_table =pd.DataFrame(list(zip(model_dict.values(), result_acc.values(), result_auc.values())), \
                  columns=['Model', 'Accuracy_rate','Roc_auc_rate'])
    return performance_table

def graph(model, X_train, y_train):
    obb = []
    est = list(range(5, 200, 5))
    for i in tqdm(est):
        random_forest = model(n_estimators=i, criterion='entropy', random_state=11, oob_score=True, n_jobs=-1, \
                           max_depth=25, min_samples_leaf=80, min_samples_split=3,)
        random_forest.fit(X_train, y_train)
        obb.append(random_forest.oob_score_)
    display('max oob {} and number of estimators {}'.format(max(obb), est[np.argmax(obb)]))
    plt.plot(est, obb)
    plt.title('model')
    plt.xlabel('number of estimators')
    plt.ylabel('oob score')
    plt.show();
    
    graph(RandomForestClassifier, X_train, y_train)

    ''' Build graph for ROC_AUC '''

    fpr, tpr, threshold = roc_curve(y_test, models[1].predict_proba(X_test)[:,1])
                                            
    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        text=threshold,
        fill='tozeroy',
        name='ROC Curve')

    trace1 = go.Scatter(
        x=[0,1],
        y=[0,1],
        line={'color': 'red', 'width': 1, 'dash': 'dash'},
        name='Baseline')

    data = [trace0, trace1]

    layout = go.Layout(
        title='ROC Curve',
        xaxis={'title': 'False Positive Rate'},
        yaxis={'title': 'True Positive Rate'})

    fig = go.Figure(data, layout)
    fig.show();

def feature_impportance():
     ''' Build bar plot of feature importance of the best model '''

def build_feature_importance(model, X_train, y_train):
    
    models = RandomForestClassifier(criterion='entropy', random_state=11, oob_score=True, n_jobs=-1, \
                           max_depth=25, min_samples_leaf=80, min_samples_split=3, n_estimators=70)
    models.fit(X_train, y_train)
    data = pd.DataFrame(models.feature_importances_, X_train.columns, columns=["feature"])
    data = data.sort_values(by='feature', ascending=False).reset_index()
    plt.figure(figsize=[6,6])
    sns.barplot(x='index', y='feature', data=data[:10], palette="Blues_d")
    plt.title('Feature inportance of Random Forest after Grid Search')
    plt.xticks(rotation=45)
    plt.show();
    
    plt.savefig.build_feature_importance(RandomForestClassifier, X_train, y_train)

















