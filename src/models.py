from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_models(config):
    """
    Создаёт модели, которые включены в config.yaml.

    """

    models = {}

    model_config = config["models"]

    if model_config["baseline_logreg"]["enabled"]:
        params = model_config["baseline_logreg"]["params"]

        models["baseline_logreg"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**params)),
            ]
        )

    if model_config["xgboost"]["enabled"]:
        params = model_config["xgboost"]["params"]
        models["xgboost"] = XGBClassifier(**params)

    if model_config["lightgbm"]["enabled"]:
        params = model_config["lightgbm"]["params"]
        models["lightgbm"] = LGBMClassifier(**params)

    if model_config["catboost"]["enabled"]:
        params = model_config["catboost"]["params"]
        models["catboost"] = CatBoostClassifier(**params)

    return models