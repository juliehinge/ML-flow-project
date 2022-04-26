import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pickle
import mlflow

import sklearn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.feature_selection import *
from sklearn import feature_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import make_scorer

if __name__ == "__main__":

    mlflow.autolog()
    with mlflow.start_run(run_name="first run"):

        df = pd.read_json("dataset.json", orient="split")
        df = df.dropna()

        # TODO: Handle missing data


        winds = ['N', 'S', 'E', 'W', 'NE','NNE', 'SE', 'SSE', 'SW', 'SSW', 'ESE', 'ENE', 'WSW','WNW', 'NSW','NW', 'NNW' ]
        df['Direction'] = df['Direction'].apply(lambda x: winds.index(x))


        col_transformer = ColumnTransformer([('poly', PolynomialFeatures(), ['Speed'])], remainder = 'passthrough')

        X = df[["Speed","Direction"]]
        y = df["Total"]
        X_train, X_test, Y_train, Y_test = train_test_split(X,y, shuffle = False, test_size = 0.2, random_state = 42)

        class OneHot(BaseEstimator, TransformerMixin):
            def __init__(self):
                return None

            def one_hot(self, X):
                winds = ['N', 'S', 'E', 'W', 'NE','NNE', 'SE', 'SSE', 'SW', 'SSW', 'ESE', 'ENE', 'WSW','WNW', 'NSW','NW', 'NNW' ]
                for direction in winds:
                    _ = []
                    for ix, row in X.iterrows():
                        if row.loc['Direction'] == winds:
                            _.append(1)
                        else:
                            _.append(0)

                        X.loc[:,direction] = _

                        X = self.X.drop('Direction', axis=1)
                return X.values

            def fit(self, X, y = None):
                return self

            def transform(self, X, y = None):
                X_cp = X.copy()
            #   X_cp = X_cp.drop(["Source_time", "Lead_hours"], axis = 1)
                return X_cp


        top_feat = feature_selection.SelectKBest()


        pipeline_lr = Pipeline(steps=[
            ('one_hot', OneHot()), 
            ('col_trans', col_transformer),
            ('scaling', StandardScaler()),
            ('feat', feature_selection.SelectKBest(k='all')),
            ('linear_model', LinearRegression())
        ])



        pipeline_svr = Pipeline(steps=[
            ('one_hot', OneHot()), 
            #('col_trans', col_transformer),
            ('scaling', StandardScaler()),
            ('feat', feature_selection.SelectKBest(k='all')),
            ('svr', SVR())
        ])



        # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
        metrics = [
            ("MAE", mean_absolute_error, []), ("r2", r2_score, [])
            ]


        number_of_splits = 5

        #TODO: Log your parameters. What parameters are important to log?
        #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`

        for train, test in TimeSeriesSplit(number_of_splits).split(X,y): 
                pipeline_lr.fit(X.iloc[train],y.iloc[train])
                predictions_LR = pipeline_lr.predict(X.iloc[test])
                truth = y.iloc[test]

                pipeline_svr.fit(X.iloc[train],y.iloc[train])
                predictions_SVR = pipeline_svr.predict(X.iloc[test])
                truth = y.iloc[test]

                from matplotlib import pyplot as plt 
                plt.plot(truth.index , truth.values , label="Truth") 
                plt.plot(truth.index, predictions_LR, label="Predictions for LR")
                plt.title("Linear Regression") 
                plt.show()

                plt.plot(truth.index , truth.values , label="Truth") 
                plt.plot(truth.index, predictions_SVR, label="Predictions for SVR") 
                plt.title("SVR") 
                plt.show()

                for name, func, scores in metrics: 
                    score_LR = func(truth, predictions_LR) 
                    scores.append(score_LR)

                for name, func, scores in metrics: 
                    score_SVR = func(truth, predictions_SVR) 
                    scores.append(score_SVR)


        scorer = make_scorer(mean_squared_error, greater_is_better=False)


        param_grid_lr = {
        "col_trans__poly__degree" : [1,2,3,4,5],
        "col_trans__poly__interaction_only" :[True, False]
        }


        grid_lr = GridSearchCV(pipeline_lr, param_grid = param_grid_lr, scoring=scorer,
                    cv=TimeSeriesSplit(n_splits=int(X.shape[0]/((24*6)/3))).split(X_train,
            Y_train), verbose=4, n_jobs=-1, refit=True, return_train_score=True)

        #grid_lr.fit(X_train, Y_train)

        parameters = [{'svr__kernel': ['rbf'],'svr__gamma':[0.0001, 0.0005,  0.001, 0.005,  0.01, 0.05, 1, 5, 10]}]

        grid_svr = GridSearchCV(pipeline_svr, param_grid = parameters, scoring=scorer,
                    cv=TimeSeriesSplit(n_splits=int(X.shape[0]/((24*6)/3))).split(X_train,
            Y_train), verbose=4, n_jobs=-1, refit=True, return_train_score=True)

        #grid_svr.fit(X_train, Y_train)

        for name, _, scores in metrics:
                # NOTE: Here we just log the mean of the scores. 
                # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            #mlflow.log_metric(f"mean_{name}", mean_score)



        selectK_mask = pipeline_lr['feat'].get_support()    
        selectK_mask = pipeline_svr['feat'].get_support()



        # List of pipelines for ease of iteration
        grids = [grid_lr, grid_svr]

        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'Linear Regression', 1: 'SVR'}

        print('Performing model optimizations...')
        best_acc = 0.0
        best_clf = 0
        best_gs = ''
        for idx, gs in enumerate(grids):
            print('\nEstimator: %s' % grid_dict[idx])	
            # Fit grid search	
            gs.fit(X_train, Y_train)

            best_params = gs.best_estimator_

            print('Score: %s' %best_params.score(X_test,Y_test))

            print('Best params: %s' % gs.best_params_)

            #print('Best training accuracy: %.3f' % gs.best_score_)

            # Predict on test data with best params
            #Y_pred = gs.predict(X_test)

            # Test data accuracy of model with best params
            #print(mean_squared_error(Y_test, Y_pred))
            # Track best (highest test accuracy) model
            if best_params.score(X_test,Y_test) > best_acc:
                best_acc = best_params.score(X_test,Y_test)
                best_gs = gs
                best_clf = idx
        print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

