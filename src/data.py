import pandas as pd

TRAIN_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}

TEST_COLUMNS = TRAIN_COLUMNS - {"Survived"}


def load_raw_data(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    check_columns(train_df, TRAIN_COLUMNS, "train")
    check_columns(test_df, TEST_COLUMNS, "test")

    return train_df, test_df


def check_columns(df, required_columns, name):

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing_columns)}"
        )