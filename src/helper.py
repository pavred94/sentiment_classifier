import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from numpy.typing import NDArray
from transformers import BertTokenizer

import constants


def reverse_label_encoding(arr: NDArray[np.int_], label_map: dict[int, str] = None) -> NDArray[str]:
    """
    Assigned human-readable labels to integer labels
    :param arr: Encoded labels
    :param label_map: Mapping from integer labels to human-readable labels
    :return: Numpy array of human-readable labels
    """
    # Default label mapping
    if label_map is None:
        label_map = ({0: "Negative", 1: "Neutral", 2: "Positive"})
    keys = np.array(list(label_map.keys()), dtype=int)
    values = np.array(list(label_map.values()))
    out_arr = np.zeros_like(arr, dtype=values.dtype)
    for key, val in zip(keys, values):
        out_arr[arr == key] = val
    return out_arr


class SentimentDataset(Dataset):
    """ A custom dataset for handling sentiment data."""

    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        sentence = self.text.iloc[idx]
        label = self.labels.iloc[idx]
        return {"sentence": sentence,
                "label": label}

    def __len__(self):
        return len(self.labels)


class _LSTMClassifier(nn.Module):
    """
    Private class.
    Define multi-classification Bidirectional LSTM neural network.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        output = self.fc(hidden)

        return output


class _Singleton(type):
    """
    Private class.
    A metaclass for singleton pattern.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LSTMToolkit(metaclass=_Singleton):
    """ Singleton class for access to BiLSTM classifier and BERT tokenizer. """

    def __init__(self):
        # Define tokenizer
        pre_trained_model_name = 'bert-base-cased'
        self.__bert_tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        self.__bert_tokenizer.model_max_length = sys.maxsize
        # Create classifier
        self._model = _LSTMClassifier(vocab_size=len(self.__bert_tokenizer),
                                      embedding_dim=constants.EMBEDDING_DIM,
                                      hidden_dim=constants.HIDDEN_DIM,
                                      n_layers=constants.N_LAYERS,
                                      bidirectional=True,
                                      dropout=constants.DROPOUT,
                                      output_dim=constants.OUTPUT_DIM)

    @property
    def model(self):
        return self._model

    def encode_text(self, text):
        """
        Tokenizes text and encodes it.
        """
        return self.__bert_tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=False,
            return_length=True,
            return_tensors='pt',
        )
