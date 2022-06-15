import pandas as pd
from torch.utils.data import Dataset
from copy import copy
from decorators import timer
import torch

def prepare_input(cfg, text):
  tokenizer = cfg.tokenizer
  inputs = tokenizer(text,
                    add_special_tokens=True,
                    max_length=cfg.max_len,
                    truncation = True,
                    padding="max_length")
  for k, v in inputs.items():
    inputs[k] = torch.tensor(v, dtype=torch.long)
  return inputs

class HateSpeechDataset(Dataset):
    @timer
    def __init__(self, cfg, dataframe, preprocessor = None) -> None:
        super().__init__()
        if preprocessor is not None:
            print("Preprocessing")
            dataframe["origin_text"] = copy(dataframe['free_text'])
            dataframe['free_text'] = dataframe['free_text'].apply(lambda x: preprocessor(x, cfg.model_path))
        self.tokenizer = cfg.tokenizer
        self.free_texts = dataframe['free_text'].tolist()
        self.labels = dataframe['label_id'].tolist()
        self.cfg = cfg

    def  __len__(self):
        return len(self.free_texts)

    def __getitem__(self, index):
        free_text = self.free_texts[index]
        label = self.labels[index]
        inputs = prepare_input(self.cfg, free_text)
        return inputs, torch.tensor(label, dtype=torch.long)