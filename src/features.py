from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


TITLE_MAP = {
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Dr": "Mr",
    "Major": "Mr",
    "Lady": "Mrs",
    "Countess": "Mrs",
    "Jonkheer": "Other",
    "Col": "Other",
    "Rev": "Other",
    "Capt": "Mr",
    "Sir": "Mr",
    "Don": "Mr",
    "Dona": "Mrs",
}


@dataclass
class FeatureBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    test_ids: pd.Series
    train_df_processed: pd.DataFrame
    test_df_processed: pd.DataFrame


def prepare_feature_bundle(train_df, test_df, config):
    """
    Готовит train и test данные для обучения модели.

    Возвращает FeatureBundle:
    - X_train: признаки для обучения
    - y_train: целевая переменная
    - X_test: признаки для Kaggle/test
    - feature_names: названия итоговых признаков
    - test_ids: PassengerId из test
    - train_df_processed: обработанный train
    - test_df_processed: обработанный test
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = add_new_features(train_df, config)
    test_df = add_new_features(test_df, config)

    age_by_title = get_age_by_title(train_df)
    age_median = train_df["Age"].median()
    fare_median = train_df["Fare"].median()

    train_df["Age"] = fill_missing_age(train_df, age_by_title, age_median)
    test_df["Age"] = fill_missing_age(test_df, age_by_title, age_median)

    train_df["Fare"] = train_df["Fare"].fillna(fare_median)
    test_df["Fare"] = test_df["Fare"].fillna(fare_median)

    categorical_cols = config["features"]["categorical_cols"]
    numeric_cols = config["features"]["numeric_cols"]
    target_col = config["data"]["target_col"]
    id_col = config["data"]["id_col"]

    train_numeric = train_df[numeric_cols].astype(float).to_numpy()
    test_numeric = test_df[numeric_cols].astype(float).to_numpy()

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    train_categorical = encoder.fit_transform(train_df[categorical_cols])
    test_categorical = encoder.transform(test_df[categorical_cols])

    X_train = np.hstack([train_numeric, train_categorical])
    X_test = np.hstack([test_numeric, test_categorical])

    categorical_feature_names = list(
        encoder.get_feature_names_out(categorical_cols)
    )

    feature_names = numeric_cols + categorical_feature_names

    return FeatureBundle(
        X_train=X_train,
        y_train=train_df[target_col].to_numpy(),
        X_test=X_test,
        feature_names=feature_names,
        test_ids=test_df[id_col],
        train_df_processed=train_df,
        test_df_processed=test_df,
    )


def add_new_features(df, config):
    """
    Добавляет новые признаки в датафрейм.
    """

    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"] = df["Initial"].replace(TITLE_MAP)

    df["Embarked"] = df["Embarked"].fillna(
        config["features"]["fill_embarked_with"]
    )

    df["Cabin"] = df["Cabin"].fillna(
        config["features"]["fill_cabin_with"]
    )

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    return df


def get_age_by_title(train_df):
    """
    Считает медианный возраст для каждого титула.

    """

    return train_df.groupby("Initial")["Age"].median().to_dict()


def fill_missing_age(df, age_by_title, fallback_age):
    """
    Заполняет пропуски в Age.

    """

    age = df["Age"].copy()
    missing_age_mask = age.isna()

    age.loc[missing_age_mask] = (
        df.loc[missing_age_mask, "Initial"]
        .map(age_by_title)
        .fillna(fallback_age)
    )

    return age