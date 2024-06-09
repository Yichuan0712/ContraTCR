import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd


class miniDataset(Dataset):
    def __init__(self, dataframe, n_pos, n_neg):
        """
        Initializes the dataset object.
        :param dataframe: A DataFrame containing the data to be used in this dataset.
        """
        # Assuming the last column of dataframe is the label and the rest are features
        self.features = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the feature tensor and label tensor at the specified index.
        :param idx: Index of the data point to retrieve.
        :return: A tuple containing the feature tensor and label tensor.
        """
        return self.features[idx], self.labels[idx]


def get_dataloader(configs, valid_fold_index, test_fold_index):
    if configs.dataset == "mini":
        def get_dataframe_mini(fold_index):
            # Load the data from CSV files
            dataframe = pd.read_csv(f'./dataset/mini/minifold_{fold_index}.csv')
            return dataframe

        # Load validation and test data
        valid_data = get_dataframe_mini(valid_fold_index)
        test_data = get_dataframe_mini(test_fold_index)

        # Combine and shuffle remaining data for training
        train_data = pd.concat([get_dataframe_mini(i) for i in range(5) if i not in [valid_fold_index, test_fold_index]],
                               ignore_index=True)

        print([i for i in range(5) if i not in [valid_fold_index, test_fold_index]])

        train_data = train_data.sample(frac=1).reset_index(drop=True)
        # return all rows, avoid inserting the old index as a column in the new DataFrame

        # Create datasets
        train_dataset = miniDataset(train_data)
        valid_dataset = miniDataset(valid_data)
        test_dataset = miniDataset(test_data)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)
        # Valid和Test应该怎么写?

        return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    else:
        raise ValueError("Wrong dataset specified.")
