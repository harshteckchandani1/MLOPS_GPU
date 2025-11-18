import torch
from torch import nn, optim
from model import SimpleNN
from data import get_data
import mlflow

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

mlflow.set_experiment("mlops-gpu-demo")

def train_model(epochs=3):
    train_loader, test_loader = get_data()

    model = SimpleNN().to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)

        for epoch in range(epochs):
            model.train()
            running_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            mlflow.log_metric("loss", avg_loss, step=epoch)

        # save model
        torch.save(model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
