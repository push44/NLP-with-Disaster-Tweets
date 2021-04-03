import pandas as pd
import config
import model_dispatcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
import pickle

def count_vectorizer(text, save=False):
    vectorizer = CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None,
        max_features=1000,
        max_df=0.9,
        min_df=0.05
    )
    vectorizer.fit(text)

    if save==True:
        with open(f"{config.MODEL_PATH}/count_vec.pickle", "wb") as f:
            pickle.dump(vectorizer, f)

    return vectorizer

def tfidf_vectorizer(text, save=False):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)

    if save==True:
        with open(f"{config.MODEL_PATH}/tfidf_vec.pickle", "wb") as f:
            pickle.dump(vectorizer, f)

    return vectorizer


def train_cv(df, model_name, vectorization_func):

    for fold in range(config.N_FOLDS):
        df_train = df[df["kfold"] != fold]
        df_valid = df[df["kfold"] == fold]

        X_train, y_train = df_train["text"], df_train["target"]
        X_valid, y_valid = df_valid["text"], df_valid["target"]
        
        vectorizer = vectorization_func(X_train)
        X_train = vectorizer.transform(X_train)
        X_valid = vectorizer.transform(X_valid)

        model = model_dispatcher.models[model_name]

        model.fit(X_train, y_train)

        for x,y,n in zip([X_train, X_valid], [y_train, y_valid], ["train", "valid"]):
            preds = model.predict(x)
            acc = metrics.accuracy_score(y, preds)
            print(f"Fold {fold} {n} accuracy: {acc}")
        print("\n")

def train(df, model_name, vectorization_func):
    X, y = df["text"], df["target"]

    vectorizer = vectorization_func(X, save=True)
    X = vectorizer.transform(X)

    model = model_dispatcher.models[model_name]
    model.fit(X, y)

    # Save the model
    with open(f"{config.MODEL_PATH}/{model_name}.pickle", "wb") as f:
        pickle.dump(model, f)



if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TRAIN_FILE)
    model_name = "logistic_reg"

    train_cv(df, model_name, tfidf_vectorizer)
    #train(df, model_name, tfidf_vectorizer)
