import csv
import os
import re

from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Set, Optional, Any, Type
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
import random
from sklearn.tree import plot_tree
import json
from sklearn.metrics import r2_score
import time


ALL_X_FEATURES: List[str] = ['max_rate', 'min_rate', 'mean_rate', 'std_rate', 'max_size', 'min_size', 'mean_size', 'std_size']  # FOR ALL EXCEPT SIMPLE TREE
ALL_Y_FEATURES: List[str] = ['max_served_adapters', 'max_total_throughput', 'max_adapter_slots']


def smape(y_true: List[float], y_pred: List[float]):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    output = float(100 * np.mean(diff / denominator))
    return output


def load_dataset(path: str, x_features, y_features) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset_x: List[List[Any]] = []
    dataset_y: List[List[Any]] = []
    paths: List[str] = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_x = []
            for feature in x_features:
                row_x.append(float(row[feature]))
            dataset_x.append(row_x)
            row_y = []
            for feature in y_features:
                row_y.append(float(row[feature]))
            dataset_y.append(row_y)
            paths.append(row['path'])
    dataset_x: np.ndarray = np.asarray(dataset_x)
    dataset_y: np.ndarray = np.asarray(dataset_y)
    return dataset_x, dataset_y, paths


def prepare_dataset(
    full_dataset_x,
    full_dataset_y,
    full_paths,
    test_dataset_x,
    test_dataset_y,
    test_paths,
    random_seed: int,
    output_path: str,
    model: str,
):
    if test_dataset_x is None:
        # do split between simulation and real data

        # split dataset into train and test
        X_train, _, y_train, _, _, test_paths = train_test_split(
            full_dataset_x,
            full_dataset_y,
            full_paths,
            test_size=0.01,
            random_state=random_seed
        )

        # store test ids
        with open(os.path.join(output_path, f'test_paths_{model}.json'), 'w') as file:
            json.dump({'test_paths': test_paths}, file, indent=4)

        X_test, y_test = None, None
    else:
        # use created split
        X_train = []
        y_train = []
        X_test = test_dataset_x
        y_test = test_dataset_y
        test_paths: Dict[str, int] = {os.path.basename(item): index for index, item in enumerate(test_paths)}
        counter_tests: int = 0
        for index in range(len(full_dataset_x)):
            if os.path.basename(full_paths[index]) in test_paths:
                counter_tests += 1
                # print(os.path.basename(full_paths[index]), f'Simulator ({full_dataset_y[index]}). Real ({test_dataset_y[test_paths[os.path.basename(full_paths[index])]]})')
            else:
                X_train.append(full_dataset_x[index])
                y_train.append(full_dataset_y[index])
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        assert counter_tests == len(test_paths)

    # print('Dataset train length', len(X_train), '. Dataset test length', len(X_test) if test_dataset_x is not None else len(test_paths))

    return X_train, y_train, X_test, y_test


def predict_dataset(
        X_train,
        y_train,
        X_test,
        y_test,
        random_seed: int,
        output_path: str,
        title: str
) -> None:
    global ALL_X_FEATURES, ALL_Y_FEATURES

    # create models
    models: Dict[str, BaseEstimator] = {}
    models_params: Dict[str, Dict[str, Any]] = {}

    # linear models
    model_name = 'LinearRegression'
    model = LinearRegression()
    models[model_name] = model
    models_params[model_name] = {}

    model_name = 'Ridge'
    model = Ridge()
    models[model_name] = model
    models_params[model_name] = {'alpha': [1, 10, 100]}

    model_name = 'BayesianRidge'
    model = BayesianRidge()
    models[model_name] = model
    models_params[model_name] = {}

    model_name = 'PLSRegression'
    model = PLSRegression()
    models[model_name] = model
    models_params[model_name] = {'n_components': [2, 4, 7]}

    # rf models
    model_name = 'RandomForestRegressor'
    model = RandomForestRegressor()
    models[model_name] = model
    models_params[model_name] = {'n_estimators': [1, 2, 4], 'max_depth': [2, 4, 8], 'max_features': [None, 'sqrt', 'log2'], 'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}

    model_name = 'RuleFitRegressor'
    model = RuleFitRegressor()
    models[model_name] = model
    models_params[model_name] = {'n_estimators': [1, 2, 4], 'tree_size': [2, 4, 8], 'max_rules': [2, 4, 8]}

    model_name = 'FIGSRegressor'
    model = FIGSRegressor()
    models[model_name] = model
    models_params[model_name] = {'max_trees': [1, 2, 4], 'max_rules': [2, 4, 8]}

    # train and test models
    best_models: Dict[str, Dict[str, BaseEstimator]] = {}
    best_models_train_scores: Dict[str, Dict[str, float]] = {}
    best_models_test_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    y_features = ALL_Y_FEATURES
    for y_feature_index, y_feature_label in enumerate(y_features):

        for model_label, model in models.items():
            # train model and find the best hyperparameters with cv
            estimator = HalvingGridSearchCV(
                estimator=model,
                param_grid=models_params[model_label],
                cv=5,
                scoring=make_scorer(smape, greater_is_better=False)
            )
            estimator.fit(X_train, y_train[:, y_feature_index])
            best_model = estimator.best_estimator_

            # save model and cv results
            if model_label not in best_models:
                best_models[model_label] = {}
                best_models_train_scores[model_label] = {}
                best_models_test_scores[model_label] = {}
            best_models[model_label][y_feature_label] = best_model
            best_models_train_scores[model_label][y_feature_label] = estimator.best_score_

            # evaluate results test
            if X_test is not None:
                init_time: float = time.perf_counter()
                y_pred_test = best_model.predict(X_test)
                prediction_time: float = ((time.perf_counter() - init_time) / len(X_test)) * 1000
                r2_result = r2_score(y_test[:, y_feature_index], y_pred_test)
                smape_result = smape(y_test[:, y_feature_index], y_pred_test)
                best_models_test_scores[model_label][y_feature_label] = {
                    'R²': r2_result,
                    'SMAPE': smape_result,
                    'time': prediction_time
                }

    # save results to csv
    rows = []
    rows.append(['target', 'model', 'r^2', 'smape', 'time'])
    for y_feature_label in y_features:
        for best_model_label in best_models:
            rows.append([
                y_feature_label,
                best_model_label,
                f'{best_models_test_scores[best_model_label][y_feature_label]['R²']:.2f}',
                f'{best_models_test_scores[best_model_label][y_feature_label]['SMAPE']:.2f}',
                f'{best_models_test_scores[best_model_label][y_feature_label]['time']:.2f}',
            ])
    with open(os.path.join(output_path, f'test_results_{title}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    # plot
    if X_test is not None:
        for y_feature_index, y_feature_label in enumerate(y_features):
            # find best model for this feature
            best_model = None
            best_model_score = None
            for model_label in best_models_test_scores.keys():
                score = best_models_test_scores[model_label][y_feature_label]['SMAPE']
                if best_model_score is None or score < best_model_score:
                    best_model_score = score
                    best_model = best_models[model_label][y_feature_label]

            # predict train and test
            y_train_feature = y_train[:, y_feature_index]
            y_test_feature = y_test[:, y_feature_index]
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # plot predictions
            plt.figure()
            plt.scatter(y_train_feature, y_pred_train, alpha=0.6, label='val data')
            plt.scatter(y_test_feature, y_pred_test, alpha=0.6, label='test data')
            plt.plot([y_train_feature.min(), y_train_feature.max()], [y_train_feature.min(), y_train_feature.max()], 'r--', label='Ideal')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'predictions_{y_feature_label}_{title}.png'))

            # plot residuals
            plt.figure()
            plt.scatter(y_pred_train, y_train_feature - y_pred_train, alpha=0.6, label='val data')
            plt.scatter(y_pred_test, y_test_feature - y_pred_test, alpha=0.6, label='test data')
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Predicted Value')
            plt.ylabel('Residual (True - Predicted)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'residuals_{y_feature_label}_{title}.png'))


def main():
    global ALL_X_FEATURES, ALL_Y_FEATURES

    # set random seed
    random_seed: int = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print(model)

        full_dataset_path: str = os.path.join('simulation_dataset', f'dataset_{model.replace(".", "")}.csv')
        full_dataset_x, full_dataset_y, full_paths = load_dataset(
            full_dataset_path,
            ALL_X_FEATURES,
            ALL_Y_FEATURES
        )
        test_dataset_path: str = os.path.join('test_dataset', f'dataset_{model.replace(".", "")}.csv')
        if os.path.exists(test_dataset_path):
            test_dataset_x, test_dataset_y, test_paths = load_dataset(
                test_dataset_path,
                ALL_X_FEATURES,
                ALL_Y_FEATURES
            )
        else:
            test_dataset_x, test_dataset_y, test_paths = None, None, None

        X_train, y_train, X_test, y_test = prepare_dataset(
            full_dataset_x,
            full_dataset_y,
            full_paths,
            test_dataset_x,
            test_dataset_y,
            test_paths,
            random_seed,
            '',
            model,
        )

        predict_dataset(
            X_train,
            y_train,
            X_test,
            y_test,
            random_seed,
            '',
            model
        )


if __name__ == '__main__':
    main()
