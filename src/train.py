import pandas as pd
import config
import model_dispatcher
from sklearn import metrics
import pickle


def vectorizer_func(text, vec_name, save=False):
    vectorizer = model_dispatcher.models[vec_name]
    vectorizer.fit(text)

    if save==True:
        with open(f"{config.MODEL_PATH}/{vec_name}.pickle", "wb") as f:
            pickle.dump(vectorizer, f)

    return vectorizer


def train_cv(df, model_name, vec_name):

    for fold in range(config.N_FOLDS):
        df_train = df[df["kfold"] != fold]
        df_valid = df[df["kfold"] == fold]

        X_train, y_train = df_train["text"], df_train["target"]
        X_valid, y_valid = df_valid["text"], df_valid["target"]
        
        vectorizer = vectorizer_func(text=X_train, vec_name=vec_name)
        X_train = vectorizer.transform(X_train)
        X_valid = vectorizer.transform(X_valid)

        model = model_dispatcher.models[model_name]

        model.fit(X_train, y_train)

        for x,y,n in zip([X_train, X_valid], [y_train, y_valid], ["train", "valid"]):
            preds = model.predict(x)
            acc = metrics.f1_score(y, preds)
            print(f"Fold {fold} {n} F1 Score: {acc}")
        print("\n")


def train(df, model_name, vec_name):
    X, y = df["text"], df["target"]

    vectorizer = vectorizer_func(text=X, vec_name=vec_name, save=True)
    X = vectorizer.transform(X)

    model = model_dispatcher.models[model_name]
    model.fit(X, y)

    # Save the model
    with open(f"{config.MODEL_PATH}/{model_name}.pickle", "wb") as f:
        pickle.dump(model, f)



if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TRAIN_FILE)

    train_cv(df, config.MODEL_NAME, config.VEC_NAME)
    train(df, config.MODEL_NAME, config.VEC_NAME)
