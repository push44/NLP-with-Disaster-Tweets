import pandas as pd
from sklearn import metrics
import torch
import numpy as np

import config
import dataset
import neural_net
import engine

def run_cv(df, fold):
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    #y_train = pd.get_dummies(train_df["target"], dtype="int64").values
    y_train = train_df["target"].values
    X_train = train_df.drop(["target", "kfold"], axis=1).values
    
    #y_valid = pd.get_dummies(valid_df["target"], dtype="int64").values
    y_valid = valid_df["target"].values
    X_valid = valid_df.drop(["target", "kfold"], axis=1).values

    train_dataset = dataset.TweetDataset(
        tweets=X_train,
        targets=y_train
    )

    valid_dataset = dataset.TweetDataset(
        tweets=X_valid,
        targets=y_valid
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = neural_net.NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    print("Training Model...")
    
    #early_stopping_counter = 0

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}\n--------------------")
        engine.train(
            train_dataloader,
            model,
            optimizer,
            loss_fn,
            device
        )

        outputs, targets = engine.evaluate(
            valid_dataloader,
            model,
            loss_fn,
            device
        )
        outputs = np.array(outputs).reshape(-1,)
        outputs = list(map(lambda pred: 1 if pred>0.5 else 0, outputs))
        valid_score = metrics.f1_score(targets, outputs)
        print(f" F1 Score: {valid_score}\n")


def run(df):
    y = df["target"].values
    X = df.drop(["target", "kfold"], axis=1).values

    train_dataset = dataset.TweetDataset(
        tweets=X,
        targets=y
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = neural_net.NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    print("Training Model...")

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}\n--------------------")
        engine.train(
            train_dataloader,
            model,
            optimizer,
            loss_fn,
            device
        )

    torch.save(model.state_dict(), f"{config.MODEL_PATH}/{config.MODEL_NAME}.pth")


if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TRAIN_FILE)

    #run_cv(df, 0)
    run(df)