# Data Source: https://amazon-reviews-2023.github.io/
# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
# https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html

import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from typing import List, Tuple, Union, Any
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import torchtext

torchtext.disable_torchtext_deprecation_warning()
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper import LSTMToolkit, reverse_label_encoding, SentimentDataset

# For LSTM model definition and BERT tokenizer
LSTM_TOOLKIT = LSTMToolkit()


def _eval_model(model: nn.Module,
                val_dataloader: DataLoader,
                scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                loss_fn: nn.CrossEntropyLoss) -> Tuple[float, float, List[float]]:
    # Predict validation set
    model.eval()
    val_acc, val_loss = 0.0, 0.0
    all_y_pred_list = []
    with torch.no_grad():
        for X_batch, y_batch, X_batch_text_len in val_dataloader:
            y_pred = model(X_batch, X_batch_text_len)
            loss = loss_fn(y_pred, y_batch)
            # Find prediction with the highest probability
            y_pred = torch.argmax(y_pred, dim=1)
            all_y_pred_list.append(y_pred.detach().cpu().numpy().squeeze())
            # Calculate evaluation metrics
            val_acc += (y_pred == y_batch).float().mean()
            val_loss += float(loss)
    avg_val_loss = val_loss / len(val_dataloader)
    scheduler.step(avg_val_loss)
    return val_acc / len(val_dataloader), avg_val_loss, all_y_pred_list


def _train_model(model: nn.Module,
                 dataloader: DataLoader,
                 optimizer: optim.Optimizer,
                 loss_fn: nn.CrossEntropyLoss,
                 epoch_idx: int,
                 n_epochs: int) -> Tuple[float, float]:
    model.train()
    training_acc, training_loss = 0.0, 0.0
    # Add progress bar
    tqdm_dataloader = tqdm(dataloader, unit="batch")
    for X_batch, y_batch, X_batch_text_len in tqdm_dataloader:
        # Zero out gradients
        optimizer.zero_grad()
        # Forward propagation
        y_pred = model(X_batch, X_batch_text_len)
        loss = loss_fn(y_pred, y_batch)
        # Backward propagation - compute gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        # Calculate model metrics
        training_acc += (torch.argmax(y_pred, dim=1) == y_batch).float().mean()
        training_loss += float(loss)
        # Set progress bar description
        tqdm_dataloader.set_description(f"Epoch [{epoch_idx + 1}/{n_epochs}]")
    return training_acc / len(dataloader), training_loss / len(dataloader)


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                loss_fn: nn.CrossEntropyLoss,
                optimizer: optim.Optimizer,
                n_epochs: int,
                model_weights_fp: str="../lstm_sentiment_classifier.pt") \
        -> Tuple[List[float], List[float], List[float], List[float], NDArray[np.int_]]:
    # Train model
    best_val_loss = np.inf
    best_y_val_pred = None
    avg_training_acc_list, avg_training_loss_list, avg_val_acc_list, avg_val_loss_list = \
        ([0] * n_epochs for i in range(4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           patience=5,
                                                           threshold=1e-4,
                                                           threshold_mode='rel',
                                                           factor=0.1)
    for i, epoch in enumerate(range(n_epochs)):
        avg_training_acc_list[i], avg_training_loss_list[i] = _train_model(model, train_dataloader,
                                                                           optimizer, loss_fn, epoch,
                                                                           n_epochs)
        avg_val_acc_list[i], avg_val_loss_list[i], y_val_pred = _eval_model(model, val_dataloader, scheduler, loss_fn)
        # Calculate gradient norm after each epoch
        gradients = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.cat(gradients).norm()
        # Record metrics for best model based on loss
        if avg_val_loss_list[i] < best_val_loss:
            print("MODEL SAVED!")
            best_val_loss = avg_val_loss_list[i]
            torch.save(model.state_dict(), model_weights_fp)
            best_y_val_pred = y_val_pred
        # Print model training & validation stats
        print(f"Training Accuracy: {avg_training_acc_list[i] * 100:.2f}% | "
              f"Training Loss: {avg_training_loss_list[i]:.4f}\n"
              f"Validation Accuracy: {avg_val_acc_list[i] * 100:.2f}% | "
              f"Validation Loss: {avg_val_loss_list[i]:.4f}\n"
              f"Gradient Norm: {grad_norm:.4f}\n"
              f"LR: {scheduler.get_last_lr()}\n")
    return (avg_training_acc_list, avg_training_loss_list,
            avg_val_acc_list, avg_val_loss_list, np.hstack(best_y_val_pred, dtype=int))


def seed_everything(seed: int):
    """
    Set seed for random number generator.
    Source: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964#file-seed_everything-py
    :param seed: Random number generator seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def open_json(filepath: str,
              max_entries: int = 100000,
              num_entries: int = 100,
              min_text_len: int = -1,
              max_text_len: int = -1) -> Union[pd.DataFrame, None]:
    if min_text_len > max_text_len > 0:
        return None
    df = pd.read_json(filepath, lines=True, nrows=max_entries)
    # Filter all entries based on criteria: min_text_len <= length(variable: text) <= max_text_len
    if min_text_len >= 0:
        df = df.loc[df.text.str.len() >= min_text_len]
    if max_text_len > 0:
        df = df.loc[df.text.str.len() <= max_text_len]
    # Bin "rating" variable (Rating -> Binned rating):
    #   1,2 -> Negative (0); 3 -> Neutral (1); 4,5 -> Positive (2)
    df['binned_rating'] = df['rating'].map({1: 0, 2: 0,
                                            3: 1,
                                            4: 2, 5: 2})
    # Get number of unique ratings
    num_labels = df.binned_rating.nunique()
    entries_per_label = num_entries // num_labels
    return pd.concat([df.loc[df.binned_rating.isin([curr_rating])].sample(entries_per_label, replace=False)
                      for curr_rating in df.binned_rating.unique()])


def my_collate(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor([i["label"] for i in batch], dtype=torch.long)
    feature_text = [i["sentence"] for i in batch]
    # Tokenize
    encoding = LSTM_TOOLKIT.encode_text(feature_text)
    # Sort based on descending text lengths pre-padding
    text_lengths = encoding["length"]
    sorted_indices = sorted(range(len(text_lengths)), key=lambda x: text_lengths[x], reverse=True)
    tokenized_text = encoding["input_ids"][sorted_indices]
    text_lengths = text_lengths[sorted_indices]
    labels = labels[sorted_indices]
    return tokenized_text, labels, text_lengths


def gen_eval_line_plot(training_list: List[float],
                       val_list: List[float],
                       title: str,
                       y_label: str,
                       x_label: str = "Epochs") -> None:
    """
    Generates two line plots: training and validation.
    """
    plt.plot(training_list, label="Training")
    plt.plot(val_list, label="Validation")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    seed_everything(123)

    # Import data
    data_df = open_json(filepath=os.path.join(os.getcwd(), "Movies_and_TV.jsonl"),
                        max_entries=200000,
                        num_entries=300,
                        min_text_len=25)[["binned_rating", "text"]]

    print(f"Number of missing values: {data_df.isnull().sum().sum()}")

    # Separate into features and labels
    features = data_df["text"]
    labels = data_df["binned_rating"]

    # See category balance of response variable
    print(f"Breakdown of response variable: {Counter(labels)}")

    # Generate 70/30 train/validation set split
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.30, random_state=123)

    # Load data into dataloaders
    train_ds = SentimentDataset(text=X_train, labels=y_train)
    val_ds = SentimentDataset(text=X_val, labels=y_val)

    train_dataloader = DataLoader(train_ds,
                                  batch_size=16,
                                  collate_fn=my_collate,
                                  shuffle=True)
    val_dataloader = DataLoader(val_ds,
                                batch_size=128,
                                collate_fn=my_collate,
                                shuffle=False)

    # Train model
    model = LSTM_TOOLKIT.model

    # Define loss function & back-propagation method
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    avg_training_acc_list, avg_training_loss_list, avg_val_acc_list, avg_val_loss_list, y_val_pred = \
        train_model(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    n_epochs=15,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    model_weights_fp="../lstm_sentiment_classifier_TEST.pt")

    # Evaluate model

    # Plots
    # Loss
    gen_eval_line_plot(training_list=avg_training_loss_list,
                       val_list=avg_val_loss_list,
                       title="Cross Entropy: Training vs. Validation",
                       y_label="Cross Entropy")
    # Accuracy
    gen_eval_line_plot(training_list=avg_training_acc_list,
                       val_list=avg_val_acc_list,
                       title="Accuracy: Training vs. Validation",
                       y_label="Accuracy")

    # Confusion matrix
    y_val_pred = reverse_label_encoding(y_val_pred.squeeze())
    y_val = reverse_label_encoding(val_ds.labels.squeeze())

    print(pd.crosstab(y_val,
                      y_val_pred,
                      rownames=['Actual Rating'],
                      colnames=['Predicted Rating']))

    print(classification_report(y_val, y_val_pred))


if __name__ == '__main__':
    main()
