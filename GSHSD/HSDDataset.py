import pandas as pd
from torch.utils.data import Dataset
from decorators import timer, debug
from copy import copy

class ViHSDData(Dataset):
    @timer
    def __init__(self, path,
                 utterance_feild,
                 label_feild,
                 text_preprocessor=None
                ):
        super(ViHSDData, self).__init__()
        self.label_feild = label_feild
        self.text_preprocessor = text_preprocessor
        self.utterance_feild = utterance_feild
        self.label_feild = label_feild

        self.data_df = pd.read_csv(path)
        if self.text_preprocessor is not None:
            print("Preprocessing")
            self.data_df["origin_"+self.utterance_feild] = copy(self.data_df[self.utterance_feild])
            self.data_df[self.utterance_feild] = self.data_df[self.utterance_feild].apply(self.text_preprocessor)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row_idx_data = self.data_df.iloc[idx]
        utterance, hate_label = row_idx_data[self.utterance_feild], row_idx_data[self.label_feild]
        return utterance, hate_label


if __name__== "__main__":
    
    def proprocess(x):
        x = str(x)
        return x.lower().strip()
    
    data = ViHSDData("GSHSD/data/vihsd/train.csv", 
                     utterance_feild = "free_text", 
                     label_feild="label_id", 
                     text_preprocessor=None)
                     
    for i, data in enumerate(data):
        print(data)
        if i == 2:
            break
