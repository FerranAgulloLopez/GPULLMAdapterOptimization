import time
import joblib
from typing import List, Dict, Any, Optional
import numpy as np
import os

from benchmarks.lora.train_ml import ALL_X_FEATURES, REG_Y_FEATURES, CLASS_Y_FEATURES


CACHED_MODELS_REG = None
CACHED_MODELS_CLASS = None
def predict_digital_twin(
        regression: bool,
        model_path: str,
        x_features: Dict[str, Any],
        y_features_to_predict: Optional[List[str]] = None,
) -> Dict[str, Any]:
    global CACHED_MODELS_REG, CACHED_MODELS_CLASS

    if regression:
        y_features = y_features_to_predict if y_features_to_predict is not None else REG_Y_FEATURES
    else:
        y_features = y_features_to_predict if y_features_to_predict is not None else CLASS_Y_FEATURES

    # load models
    if regression:
        if CACHED_MODELS_REG is None:
            models = []
            for y_feature_label in y_features:
                models.append(joblib.load(os.path.join(model_path, f'best_model_{y_feature_label}.pkl')))
            CACHED_MODELS_REG = models
        else:
            models = CACHED_MODELS_REG
    else:
        if CACHED_MODELS_CLASS is None:
            models = []
            for y_feature_label in y_features:
                models.append(joblib.load(os.path.join(model_path, f'best_model_{y_feature_label}.pkl')))
            CACHED_MODELS_CLASS = models
        else:
            models = CACHED_MODELS_CLASS

    # prepare input data
    input_array: np.ndarray = np.asarray([x_features[x_feature_label] for x_feature_label in ALL_X_FEATURES]).reshape(1, -1)

    # predict output data
    init_time: float = time.perf_counter()
    output: Dict[str, Any] = {}
    for index, y_feature_label in enumerate(y_features):
        output[y_feature_label] = models[index].predict(input_array)
    output['duration'] = time.perf_counter() - init_time

    return output
