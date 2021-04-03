import pandas as pd
from sklearn import model_selection
import config

def create_folds(dataframe):
    dataframe["kfold"] = -1
    dataframe = dataframe.sample(frac=1, random_state=5).reset_index(drop=True)
    y = dataframe["target"].values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=dataframe, y=y)):
        dataframe.loc[v_, "kfold"] = f

    return dataframe


if __name__ == "__main__":
    train_df = pd.read_csv(config.ARCHIVE_TRAIN_FILE)
    train_df = create_folds(train_df)
    train_df.to_csv(config.DEV_TRAIN_FILE, index=False)

    test_df = pd.read_csv(config.ARCHIVE_TEST_FILE)
    test_df.to_csv(config.DEV_TEST_FILE, index=False)