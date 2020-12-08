from pickle import NONE
from sklearn.model_selection import train_test_split

root_dir = "~/repository/exercise/ridge_reg"


def train_val_test_split(
    df, val=0.15, test=0.1, shuffle=False, stratify_cols=None, random_state=None
):
    """
    training validation test data split
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test,
        shuffle=shuffle,
        stratify=df[stratify_cols],
        random_state=random_state,
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=val,
        shuffle=shuffle,
        stratify=df_train[stratify_cols],
        random_state=random_state,
    )
    return df_train, df_val, df_test
