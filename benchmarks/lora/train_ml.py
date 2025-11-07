import argparse
import csv
import json
import os
import random
import time
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score, f1_score

ALL_X_FEATURES: List[str] = ['sum_rate', 'std_rate', 'max_size', 'mean_size', 'std_size', 'adapter_slots', 'served_adapters']
ALL_Y_FEATURES: List[str] = ['total_throughput', 'starvation', 'itl', 'ttft']
REG_Y_FEATURES: List[str] = ['total_throughput', 'itl', 'ttft']
CLASS_Y_FEATURES: List[str] = ['starvation']


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


def train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        random_seed: int,
        output_path: str,
        import_statement: str,
        class_name: str,
        parameters_to_test: dict,
        y_features: List[str],
        classification: bool,
) -> None:
    # define model
    model_name = os.path.basename(output_path)
    exec(import_statement)
    model = eval(class_name)()

    # prepare csv header
    if not classification:
        row = ['target', 'model', 'r_2', 'smape', 'time']
    else:
        row = ['target', 'model', 'accuracy', 'f1_macro', 'time']
    with open(os.path.join(output_path, f'test_results.csv'), mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # train and test model
    for y_feature_index, y_feature_label in enumerate(y_features):
        print(f'\n\nTraining to predict {y_feature_label}')

        # train model and find the best hyperparameters with cv
        scoring = make_scorer(smape, greater_is_better=False) if not classification else 'f1_macro'
        estimator = HalvingGridSearchCV(
            estimator=model,
            param_grid=parameters_to_test,
            cv=5,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )
        estimator.fit(X_train, y_train[:, y_feature_index])

        # extract test results of best model
        init_time: float = time.perf_counter()
        y_pred_test = estimator.best_estimator_.predict(X_test)
        prediction_time: float = ((time.perf_counter() - init_time) / len(X_test)) * 1000
        if not classification:
            r2_result = r2_score(y_test[:, y_feature_index], y_pred_test)
            smape_result = smape(y_test[:, y_feature_index], y_pred_test)
            print(f'Training finished. Obtained results in test: R2 -> {r2_result}; SMAPE -> {smape_result}; time -> {prediction_time}')
        else:
            accuracy = accuracy_score(y_test[:, y_feature_index], y_pred_test)
            f1_macro = f1_score(y_test[:, y_feature_index], y_pred_test, average='macro')
            print(f'Training finished. Obtained results in test: Accuracy -> {accuracy}; F1-macro -> {f1_macro}; time -> {prediction_time}')

        # save test results to csv
        if not classification:
            row = [
                y_feature_label,
                model_name,
                f'{r2_result:.2f}',
                f'{smape_result:.2f}',
                f'{prediction_time:.2f}',
            ]
        else:
            row = [
                y_feature_label,
                model_name,
                f'{accuracy:.2f}',
                f'{f1_macro:.2f}',
                f'{prediction_time:.2f}',
            ]
        with open(os.path.join(output_path, f'test_results.csv'), mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # save best model
        joblib.dump(estimator.best_estimator_, os.path.join(output_path, f'best_model_{y_feature_label}.pkl'))

        # save best model params
        with open(os.path.join(output_path, f'best_params_{y_feature_label}.json'), 'w') as file:
            json.dump(estimator.best_params_, file, indent=4)


def main(
        output_path: str,
        train_dataset_path: str,
        test_dataset_path: str,
        import_statement: str,
        class_name: str,
        parameters_to_test: dict,
        predict_classification_features: bool,
):
    global ALL_X_FEATURES, REG_Y_FEATURES, CLASS_Y_FEATURES

    # set random seed
    random_seed: int = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    # load datasets
    if predict_classification_features:
        y_features = CLASS_Y_FEATURES
    else:
        y_features = REG_Y_FEATURES
    train_dataset_x, train_dataset_y, _ = load_dataset(
        train_dataset_path,
        ALL_X_FEATURES,
        y_features
    )
    test_dataset_x, test_dataset_y, _ = load_dataset(
        test_dataset_path,
        ALL_X_FEATURES,
        y_features
    )

    # transform to numpy
    X_train = np.asarray(train_dataset_x)
    y_train = np.asarray(train_dataset_y)
    X_test = np.asarray(test_dataset_x)
    y_test = np.asarray(test_dataset_y)

    # run training
    train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_seed=random_seed,
        output_path=output_path,
        import_statement=import_statement,
        class_name=class_name,
        parameters_to_test=parameters_to_test,
        y_features=y_features,
        classification=predict_classification_features,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for training ML models to replicate Digital Twin results')
    parser.add_argument('--output-path', type=str, help='Directory to store results', required=True)
    parser.add_argument('--train-dataset-path', type=str, help='Training dataset to use', required=True)
    parser.add_argument('--test-dataset-path', type=str, help='Testing dataset to use', required=True)
    parser.add_argument('--import-statement', type=str, help='Import statement to use to import model', required=True)
    parser.add_argument('--class-name', type=str, help='Class name of the model to instantiate', required=True)
    parser.add_argument('--parameters-to-test', type=json.loads, help='Arguments to test in hyperparameter search', required=True)
    parser.add_argument('--predict-classification-features', default=False, action='store_true', help='Predict the classification features instead of the regression ones')
    args = parser.parse_args()
    main(
        output_path=args.output_path,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        import_statement=args.import_statement,
        class_name=args.class_name,
        parameters_to_test=args.parameters_to_test,
        predict_classification_features=args.predict_classification_features,
    )
