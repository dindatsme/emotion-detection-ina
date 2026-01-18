from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd

def multilabel_stratified_split(df: pd.DataFrame, labels: list, train_size: float = 0.8, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Splits the DataFrame into training and testing sets using multilabel stratified shuffle split.

    Parameters:
    - df: pd.DataFrame - The input DataFrame containing features and labels.
    - labels: list - List of column names that represent the labels.
    - test_size: float - Proportion of the dataset to include in the test split.
    - val_size: float - Proportion of the dataset to include in the validation split.
    - train_size: float - Proportion of the dataset to include in the training split.
    - random_state: int - Random seed for reproducibility.

    Returns:
    - df_train: pd.DataFrame - Training set.
    - df_val: pd.DataFrame - Validation set.
    - df_test: pd.DataFrame - Testing set.
    """
    # 1. First split: Train vs Rest (Validation + Test)
    test_val_ratio = str(val_size + test_size)
    msss_1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
    
    X = df.index.values
    y = df[labels].values
    
    train_idx, rest_idx = next(msss_1.split(X, y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_rest = df.iloc[rest_idx].reset_index(drop=True)
    
    # 2. Second split: Validation vs Test from the 'Rest' set
    # Calculate the ratio for the second split (e.g., 0.5 if Val=10% and Test=10%)
    split_ratio_2 = str(test_size / (val_size + test_size))
    msss_2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=split_ratio_2, random_state=random_state)
    
    X_rest = df_rest.index.values
    y_rest = df_rest[labels].values
    
    val_idx, test_idx = next(msss_2.split(X_rest, y_rest))
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)
    df_test = df_rest.iloc[test_idx].reset_index(drop=True)
    
    return df_train, df_val, df_test