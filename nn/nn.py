import torch
import torch.optim
from tqdm import tqdm


def train(device, model, loss_function, optimizer, scheduler, data, epochs=10):
    for epoch in tqdm(range(epochs), desc=f"Training over {epochs} epochs"):
        running_loss = float(0)
        model.train()
        for batch in data:
            image, label = batch["flair"].to(device), batch["seg"].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, label)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * image.size(0)
        epoch_loss = running_loss / (len(data.dataset))
        scheduler.step()
        print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
        print("----------------------------------------")
    return model
