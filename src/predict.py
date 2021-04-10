import pandas as pd
import numpy as np
import config
import torch

import neural_net

if __name__ == "__main__":
    df = pd.read_csv(config.DEV_TEST_FILE)
    submission_df = pd.read_csv(config.ARCHIVE_SUBMISSION_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = neural_net.NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/{config.MODEL_NAME}.pth"))
    model.eval()

    X = torch.tensor(df.values)
    X = X.to(device, dtype=torch.float)
    predictions = model(X)

    predictions = predictions.reshape(-1,)
    predictions = list(map(lambda pred: 1 if pred>0.5 else 0, predictions))

    submission_df["target"] = predictions
    submission_df.to_csv(config.ARCHIVE_SUBMISSION_FILE, index=False)