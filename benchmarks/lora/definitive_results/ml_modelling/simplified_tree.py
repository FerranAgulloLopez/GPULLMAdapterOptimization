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


ALL_X_FEATURES: List[str] = ['max_rate', 'min_rate', 'mean_rate', 'std_rate', 'max_size', 'min_size', 'mean_size']  # FOR SIMPLE TREE
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


def collapse_equal_leaves(tree):
    """
    Collapse internal nodes if both children are leaves
    and predict the same value.
    """
    def _prune_node(node_id):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # If it's already a leaf, nothing to do
        if left == -1 and right == -1:
            return

        # Recurse on children first
        if left != -1:
            _prune_node(left)
        if right != -1:
            _prune_node(right)

        # After recursion, check if both children are leaves
        if (tree.children_left[left] == -1 and
            tree.children_left[right] == -1):

            # Compare predictions
            left_val = tree.value[left][0,0]
            right_val = tree.value[right][0,0]

            if np.isclose(left_val, right_val):
                # Make current node a leaf
                tree.children_left[node_id] = -1
                tree.children_right[node_id] = -1
                tree.feature[node_id] = -2   # convention for "no split"
                tree.threshold[node_id] = -2.0
                tree.value[node_id][0,0] = left_val  # keep that prediction

    _prune_node(0)  # start at root


def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left child
            recurse(tree_.children_left[node], path + [f"{name} <= {threshold:.3f}"])
            # Right child
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.3f}"])
        else:
            # Leaf node
            paths.append((path, tree_.value[node][0][0]))

    recurse(0, [])
    return paths


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

    # simple rules
    model_name = 'RandomForestRegressorSimple'
    model = RandomForestRegressor()
    models[model_name] = model
    models_params[model_name] = {'n_estimators': [1], 'max_depth': [2, 4, 6], 'max_features': [None, 'sqrt', 'log2'], 'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 'min_samples_leaf': [10], 'ccp_alpha': [0.0]}

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

            # if simple forest squeeze
            if model_label == 'RandomForestRegressorSimple':
                collapse_equal_leaves(best_model.estimators_[0].tree_)

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
    with open(os.path.join(output_path, f'test_results_{title}_simplified_tree.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    # plot
    if X_test is not None:
        # plot explainability for default random forest
        x_features = ALL_X_FEATURES
        for y_feature_label, best_model in best_models['RandomForestRegressorSimple'].items():
            # graphviz tree plots
            os.makedirs(f"output_simplified_trees/trees_{y_feature_label}_{title}", exist_ok=True)
            # Loop over estimators
            from sklearn.tree import export_graphviz
            import graphviz
            for i, estimator in enumerate(best_model.estimators_):
                dot_data = export_graphviz(
                    estimator,
                    out_file=None,
                    feature_names=x_features,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    impurity=False,
                    proportion=True,
                    node_ids=False
                )
                new_lines = []
                for line in dot_data.splitlines():
                    if 'label=<' in line:
                        label_content = re.search(r'label=<(.*)>,', line).group(1)

                        if label_content.startswith('samples = '):
                            # This is a leaf
                            new_lines.append(line)
                        else:
                            # This is an internal node
                            cleaned = re.sub(r'<br\/>samples = (.*)', '', label_content)
                            new_line = re.sub(r'label=<(.*)>,', f'label=<{cleaned}>,', line)
                            new_lines.append(new_line)
                    else:
                        new_lines.append(line)

                dot_data = "\n".join(new_lines)

                def round_floats(match):
                    return str(round(float(match.group()), 2))

                dot_data = re.sub(r'\d+\.\d+', round_floats, dot_data)
                graph = graphviz.Source(dot_data)
                graph.render(os.path.join(output_path, f'output_simplified_trees/trees_{y_feature_label}_{title}', str(i)), format="png", cleanup=True)  # saves as PNG

            # rule extraction
            all_rules = []
            for i, tree in enumerate(best_model.estimators_):
                rules = tree_to_rules(tree, x_features)
                all_rules.extend(rules)
            print(f'\n\n\n Rules for {y_feature_label} N# {len(all_rules)}')
            for path, value in all_rules:
                print(' AND '.join(path), '=>', value)


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
