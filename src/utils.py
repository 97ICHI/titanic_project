import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import RocCurveDisplay


def set_global_seed(seed):
    """
    Фиксирует случайность в Python, NumPy и PyTorch.
    Нужно, чтобы результаты экспериментов были более воспроизводимыми.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger():
    """
    Создаёт logger для вывода сообщений о ходе работы программы.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    return logging.getLogger("titanic")


def save_json(data, path):
    """
    Сохраняет словарь в JSON-файл.
    """

    json_text = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    Path(path).write_text(
        json_text,
        encoding="utf-8",
    )


def plot_roc(model, X, y, path):
    """
    Строит и сохраняет ROC-кривую модели.
    """

    RocCurveDisplay.from_estimator(
        model,
        X,
        y,
    )

    save_plot(path)


def plot_feature_importance(feature_names, values, path):
    """
    Строит и сохраняет график важности признаков.
    Показывает только топ-15 самых важных признаков.
    """

    top_count = 15

    sorted_indexes = np.argsort(values)[::-1]
    top_indexes = sorted_indexes[:top_count]

    top_feature_names = np.array(feature_names)[top_indexes]
    top_values = values[top_indexes]

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_feature_names[::-1],
        top_values[::-1],
    )

    save_plot(path)


def plot_dl_learning_curve(curve, path):
    """
    Строит и сохраняет график обучения нейросети:
    train_loss и val_loss.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        curve["train_loss"],
        label="train_loss",
    )

    plt.plot(
        curve["val_loss"],
        label="val_loss",
    )

    plt.legend()

    save_plot(path)


def save_plot(path):
    """
    Общая функция для сохранения графиков.
    """

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()