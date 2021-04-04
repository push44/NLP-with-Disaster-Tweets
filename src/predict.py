import pandas as pd
import config
import pickle

if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TEST_FILE)
    submission_df = pd.read_csv(config.ARCHIVE_SUBMISSION_FILE)

    X = df.values

    vectorization = False
    if vectorization:
        with open(f"{config.MODEL_PATH}/{config.VEC_NAME}.pickle", "rb") as f:
            vectorizer = pickle.load(f)

        X = vectorizer.transform(X)

    with open(f"{config.MODEL_PATH}/{config.MODEL_NAME}.pickle", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)
    submission_df["target"] = preds
    submission_df.to_csv(config.ARCHIVE_SUBMISSION_FILE, index=False)