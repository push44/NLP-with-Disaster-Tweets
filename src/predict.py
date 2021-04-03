import pandas as pd
import config
import pickle

if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TEST_FILE)
    submission_df = pd.read_csv(config.ARCHIVE_SUBMISSION_FILE)

    X = df["text"].values

    model_name = "naive_bayes"

    with open(f"{config.MODEL_PATH}/tfidf_vec.pickle", "rb") as f:
        vectorizer = pickle.load(f)

    with open(f"{config.MODEL_PATH}/{model_name}.pickle", "rb") as f:
        model = pickle.load(f)

    X = vectorizer.transform(X)

    preds = model.predict(X)
    submission_df["target"] = preds
