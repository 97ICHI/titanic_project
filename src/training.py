from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.dl import train_dl_model
from src.models import build_models
from src.utils import (
    plot_dl_learning_curve,
    plot_feature_importance,
    plot_roc,
    save_json,
)


def run_full_experiment(bundle, config):
    """
    Запускает полный эксперимент:

    1. Обучает классические ML-модели через cross-validation
    2. Выбирает лучшую модель
    3. Обучает лучшую модель на всём train
    4. Делает prediction для test
    5. Сохраняет результаты, submission и графики
    6. При необходимости обучает нейросеть
    7. Сохраняет summary эксперимента
    """

    create_artifacts_dir(config)

    cv = create_cv(config)

    scoring = config["cv"]["scoring"]
    primary_metric = config["cv"]["primary_metric"]

    models = build_models(config)

    results_df, best_model_name, best_metric, best_model = evaluate_models(
        models=models,
        bundle=bundle,
        cv=cv,
        scoring=scoring,
        primary_metric=primary_metric,
    )

    best_model.fit(bundle.X_train, bundle.y_train)

    test_pred = best_model.predict(bundle.X_test)

    save_results(
        results_df=results_df,
        path=config["paths"]["results_csv"],
    )

    save_submission(
        test_ids=bundle.test_ids,
        predictions=test_pred,
        path=config["paths"]["submission_csv"],
    )

    save_model_plots(
        model=best_model,
        bundle=bundle,
        config=config,
    )

    if config["models"]["dl_mlp"]["enabled"]:
        results_df = run_dl_experiment(
            bundle=bundle,
            config=config,
            results_df=results_df,
        )

        save_results(
            results_df=results_df,
            path=config["paths"]["results_csv"],
        )

    save_experiment_summary(
        best_model_name=best_model_name,
        best_metric=best_metric,
        primary_metric=primary_metric,
        bundle=bundle,
        config=config,
    )


def create_artifacts_dir(config):
    """
    Создаёт папку для сохранения результатов эксперимента.
    """

    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def create_cv(config):
    """
    Создаёт объект cross-validation.
    """

    return StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=config["cv"]["shuffle"],
        random_state=config["cv"]["random_state"],
    )


def evaluate_models(models, bundle, cv, scoring, primary_metric):
    """
    Обучает несколько ML-моделей через cross-validation.
    Возвращает:
    - таблицу результатов
    - имя лучшей модели
    - значение лучшей метрики
    - саму лучшую модель
    """

    results = []

    best_model_name = None
    best_metric = -np.inf
    best_model = None

    for model_name, model in models.items():
        row = evaluate_one_model(
            model_name=model_name,
            model=model,
            bundle=bundle,
            cv=cv,
            scoring=scoring,
        )

        results.append(row)

        current_metric = row[f"cv_{primary_metric}_mean"]

        if current_metric > best_metric:
            best_metric = current_metric
            best_model_name = model_name
            best_model = clone(model)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=f"cv_{primary_metric}_mean",
        ascending=False,
    )

    return results_df, best_model_name, best_metric, best_model


def evaluate_one_model(model_name, model, bundle, cv, scoring):
    """
    Считает cross-validation для одной модели.
    """

    scores = cross_validate(
        estimator=model,
        X=bundle.X_train,
        y=bundle.y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    row = {
        "model_name": model_name,
    }

    for metric_name in scoring:
        metric_scores = scores[f"test_{metric_name}"]

        row[f"cv_{metric_name}_mean"] = float(np.mean(metric_scores))
        row[f"cv_{metric_name}_std"] = float(np.std(metric_scores))

    return row


def save_results(results_df, path):
    """
    Сохраняет таблицу результатов моделей.
    """

    results_df.to_csv(path, index=False)


def save_submission(test_ids, predictions, path):
    """
    Создаёт и сохраняет submission-файл для Kaggle.
    """

    submission_df = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Survived": predictions.astype(int),
        }
    )

    submission_df.to_csv(path, index=False)


def save_model_plots(model, bundle, config):
    """
    Сохраняет графики для лучшей модели.
    """

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(
            feature_names=bundle.feature_names,
            values=np.asarray(model.feature_importances_),
            path=config["paths"]["importance_png"],
        )

    plot_roc(
        model=model,
        X=bundle.X_train,
        y=bundle.y_train,
        path=config["paths"]["roc_png"],
    )


def run_dl_experiment(bundle, config, results_df):
    """
    Обучает нейросеть MLP, сохраняет learning curve
    и добавляет результат нейросети в общую таблицу.

    Важно:
    для dl_mlp метрики считаются на validation split,
    а не через полноценную cross-validation.
    Поэтому mean-колонки заполняются validation-метриками,
    а std-колонки остаются NaN.
    """

    dl_result = train_dl_model(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        params=config["models"]["dl_mlp"]["params"],
        seed=config["seed"],
    )

    plot_dl_learning_curve(
        curve=dl_result.learning_curve,
        path=config["paths"]["dl_curve_png"],
    )

    dl_metrics = {
        "accuracy": dl_result.val_accuracy,
        "f1": dl_result.val_f1,
        "roc_auc": dl_result.val_roc_auc,
    }

    dl_row_data = {
        "model_name": "dl_mlp",
    }

    for metric_name in config["cv"]["scoring"]:
        dl_row_data[f"cv_{metric_name}_mean"] = float(dl_metrics.get(metric_name, np.nan))
        dl_row_data[f"cv_{metric_name}_std"] = np.nan

    dl_row = pd.DataFrame([dl_row_data])

    dl_row = dl_row.reindex(columns=results_df.columns)

    results_df = pd.concat(
        [results_df, dl_row],
        ignore_index=True,
    )

    primary_metric = config["cv"]["primary_metric"]

    results_df = results_df.sort_values(
        by=f"cv_{primary_metric}_mean",
        ascending=False,
    ).reset_index(drop=True)

    return results_df


def save_experiment_summary(
    best_model_name,
    best_metric,
    primary_metric,
    bundle,
    config,
):
    """
    Сохраняет краткое описание эксперимента в JSON.
    """

    summary = {
        "best_model": best_model_name,
        "best_primary_metric": float(best_metric),
        "primary_metric": primary_metric,
        "n_features": len(bundle.feature_names),
        "seed": config["seed"],
    }

    save_json(
        summary,
        config["paths"]["summary_json"],
    )