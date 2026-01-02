import torch
from torch.utils.data import Dataset

class NumberDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        number = self.data[index]
        input_tensor = torch.tensor(float(number))
        target_tensor = torch.tensor(float(number * 2))
        return input_tensor, target_tensor

my_data = [1, 2, 3, 4]
dataset = NumberDataset(my_data)

sample_input, sample_target = dataset[0] 

# 4. VIEW THE RESULTS
print(f"Input Tensor: {sample_input}")
print(f"Target Tensor: {sample_target}")
print(f"Dataset Length: {len(dataset)}")