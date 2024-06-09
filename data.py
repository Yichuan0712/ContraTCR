from torch.utils.data import DataLoader
import pandas as pd


def prepare_dataloader(configs, valid_fold_index, test_fold_index):
    if configs.dataset == "mini":
        def load_data_mini(fold_index):
            # Load the data from CSV files
            dataframe = pd.read_csv(f'./dataset/mini/minifold_{fold_index}.csv')
            return dataframe
        # Load validation and test data
        valid_data = load_data_mini(valid_fold_index)
        test_data = load_data_mini(test_fold_index)

        # Combine and shuffle remaining data for training
        train_data = pd.concat([load_data_mini(i) for i in range(5) if i not in [valid_fold_index, test_fold_index]],
                               ignore_index=True)
        train_data = train_data.sample(frac=1).reset_index(drop=True)

        # Create datasets
        train_dataset = miniDataset(train_data)
        valid_dataset = miniDataset(valid_data)
        test_dataset = miniDataset(test_data)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
    else:
        raise ValueError("Wrong dataset specified.")
