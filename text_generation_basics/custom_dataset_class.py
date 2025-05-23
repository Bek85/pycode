from torch.utils.data import Dataset, DataLoader
from sample_data import data
from text_gen import tokenizer, padded_input_ids, padded_attention_masks


# Create a custom dataset class including data labels
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_masks": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


# Apply the class
dataset = TextDataset(padded_input_ids, padded_attention_masks)

print(dataset[:2])
