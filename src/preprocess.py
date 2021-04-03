import pandas as pd
import config
import clean_text


if __name__ == "__main__":
    for fname in [config.DEV_TRAIN_FILE, config.DEV_TEST_FILE]:
        dataframe = pd.read_csv(fname)
        dataframe["text"] = dataframe["text"].apply(clean_text.clean())
        dataframe.to_csv(fname, index=False)
        