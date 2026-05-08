import argparse

from src.config import load_config
from src.data import load_raw_data
from src.features import prepare_feature_bundle
from src.training import run_full_experiment
from src.utils import get_logger, set_global_seed


def parse_args():
    """
    Считывает аргументы из командной строки.
    Например:
    python main.py --config configs/default.yaml
    """

    parser = argparse.ArgumentParser(
        description="Run Titanic training pipeline."
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config.",
    )

    return parser.parse_args()


def main():
    """
    Главная функция проекта.
    Запускает весь ML-пайплайн:
    загрузка конфига → загрузка данных → подготовка признаков → обучение.
    """

    args = parse_args()
    config = load_config(args.config)

    logger = get_logger()

    set_global_seed(config["seed"])

    logger.info("Loading raw data...")

    train_df, test_df = load_raw_data(
        config["paths"]["train_csv"],
        config["paths"]["test_csv"],
    )

    logger.info("Preparing features...")

    bundle = prepare_feature_bundle(
        train_df,
        test_df,
        config,
    )

    logger.info("Running training and evaluation...")

    run_full_experiment(
        bundle,
        config,
    )

    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()