from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DLRunResult:
    """
    Результат обучения нейросети.
    """

    val_accuracy: float
    val_f1: float
    val_roc_auc: float
    learning_curve: dict
    test_pred: np.ndarray


class MLP(nn.Module):
    """

    Вход:
    - количество признаков input_dim

    Выход:
    - 2 числа: логит для класса 0 и логит для класса 1
    """

    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, x):
        return self.net(x)


def train_dl_model(X_train, y_train, X_test, params, seed):
    """
    Главная функция обучения нейросети.

    Делает полный пайплайн:
    1. Делит train на train/validation
    2. Масштабирует признаки
    3. Создаёт DataLoader
    4. Создаёт модель
    5. Обучает модель
    6. Считает метрики
    7. Делает предсказания для test
    """

    torch.manual_seed(seed)


    X_tr, X_val, y_tr, y_val = split_train_validation(
        X_train,
        y_train,
        seed,
    )


    X_tr, X_val, X_test = scale_features(
        X_tr,
        X_val,
        X_test,
    )


    train_loader = create_data_loader(
        X=X_tr,
        y=y_tr,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    val_loader = create_data_loader(
        X=X_val,
        y=y_val,
        batch_size=params["batch_size"],
        shuffle=False,
    )


    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )


    train_loss_history, val_loss_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=params["epochs"],
        patience=params["patience"],
    )


    val_pred, val_prob = predict_validation(
        model=model,
        X_val=X_val,
    )


    test_pred = predict_classes(
        model=model,
        X=X_test,
    )


    return DLRunResult(
        val_accuracy=float(accuracy_score(y_val, val_pred)),
        val_f1=float(f1_score(y_val, val_pred)),
        val_roc_auc=float(roc_auc_score(y_val, val_prob)),
        learning_curve={
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
        },
        test_pred=test_pred,
    )


def split_train_validation(X_train, y_train, seed):

    return train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=seed,
    )


def scale_features(X_tr, X_val, X_test):
    """
    Масштабирует признаки через StandardScaler.

    """

    scaler = StandardScaler()

    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_tr_scaled, X_val_scaled, X_test_scaled


def create_data_loader(X, y, batch_size, shuffle):
    """
    Превращает numpy-массивы в PyTorch DataLoader.
    """

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    epochs,
    patience,
):
    """
    Обучает модель несколько эпох.

    Использует early stopping:
    если validation loss долго не улучшается,
    обучение останавливается.
    """

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_loss_history, val_loss_history


def train_one_epoch(model, loader, loss_function, optimizer):
    """
    Обучает модель одну эпоху.
    """

    model.train()

    losses = []

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_function(logits, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def evaluate_loss(model, loader, loss_function):
 
    model.eval()

    losses = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            losses.append(loss.item())

    return float(np.mean(losses))


def predict_validation(model, X_val):

    model.eval()

    X_tensor = torch.tensor(X_val, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)

        probabilities = torch.softmax(logits, dim=1)
        val_prob = probabilities[:, 1].numpy()

        val_pred = logits.argmax(dim=1).numpy()

    return val_pred, val_prob


def predict_classes(model, X):

    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        predicted_classes = logits.argmax(dim=1).numpy()

    return predicted_classes