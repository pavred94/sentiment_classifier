import sys
import numpy as np
from numpy.typing import NDArray
from transformers import BertTokenizer

from model_classes import LSTMClassifier

def get_lstm_model(vocab_size: int) -> LSTMClassifier:
    return LSTMClassifier(vocab_size=vocab_size,
                          embedding_dim=100,
                          hidden_dim=100,
                          n_layers=1,
                          bidirectional=True,
                          dropout=0.0,
                          output_dim=3)

def get_tokenizer() -> BertTokenizer:
    pre_trained_model_name = 'bert-base-cased'
    bert_tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    bert_tokenizer.model_max_length = sys.maxsize
    return bert_tokenizer

def reverse_label_encoding(arr: NDArray[np.int_], label_map=None) -> NDArray[str]:
    # Default label mapping
    if label_map is None:
        label_map = ({0: "Negative", 1: "Neutral", 2: "Positive"})
    keys = np.array(list(label_map.keys()), dtype=int)
    values = np.array(list(label_map.values()))
    out_arr = np.zeros_like(arr, dtype=values.dtype)
    for key, val in zip(keys, values):
        out_arr[arr == key] = val
    return out_arr