import torch
from torch import nn

def train(dataloader, model, optimizer, loss_fn, device):
    model.train()

    for batch, data in enumerate(dataloader):
        X = data["tweet"]
        y = data["target"]

        X = X.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)
        
        optimizer.zero_grad()

        predictions = model(X)

        loss = loss_fn(predictions.view(-1,) ,y)

        loss.backward()

        optimizer.step()


def evaluate(dataloader, model, loss_fn, device):
    model.eval()

    size = len(dataloader)
    final_predictions = []
    final_targets = []
    valid_loss = 0
    with torch.no_grad():
        for data in dataloader:
            X = data["tweet"]
            y = data["target"]

            X = X.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            predictions = model(X)

            loss = loss_fn(predictions.view(-1,), y).item()
            valid_loss += loss

            predictions = predictions.cpu().numpy().tolist()

            y = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(y)

    valid_loss/=size
    print(f"Validation Error:\n Loss: {valid_loss}")

    return final_predictions, final_targets
