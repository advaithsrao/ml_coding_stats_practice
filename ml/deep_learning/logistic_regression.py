import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt

from ml import time_ml_training
from ml.deep_learning import BaseModel
from ml.helpers import load_pytorch_dataset

display_every_n_epochs = 1

class LogisticRegressor(BaseModel):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegressor, self).__init__(n_inputs, n_outputs)
        self.input_to_h1 = nn.Linear(in_features=n_inputs, out_features=256)
        self.h1_to_h2 = nn.Linear(in_features=256, out_features=64)
        self.h2_to_h3 = nn.Linear(in_features=64, out_features=16)
        self.h3_to_outputs = nn.Linear(in_features=16, out_features=n_outputs)
    
    def forward(self, X):
        model = nn.Sequential(
            self.input_to_h1,
            self.relu,
            self.h1_to_h2,
            self.relu,
            self.h2_to_h3,
            self.relu,
            self.h3_to_outputs,
        )
        return model(X)

@time_ml_training
def train(model, train_data, num_epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch_num in range(num_epochs):
        avg_loss = []
        if epoch_num % display_every_n_epochs == 0:
            print(f'Epoch: {epoch_num}')
        for batch_X, batch_y in train_data:
            prediction = model.forward(batch_X.view(-1, 28*28))
            loss = loss_fn(prediction, batch_y)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = sum(avg_loss) / len(avg_loss)
        if epoch_num % display_every_n_epochs == 0:
            print(f'Loss: {avg_loss}')

def predict(model, test_X):
    with torch.no_grad():
        prediction = model.forward(test_X.view(-1, 28*28))
    return torch.argmax(prediction, dim=1)



if __name__ == "__main__":
    train_loader, test_loader = load_pytorch_dataset(MNIST, batch_size=64)
    model = LogisticRegressor(28*28, 10)
    train(model, train_loader, num_epochs=10)
    print('Training Done')
    
    # Predict on test data
    test_X = test_loader.dataset.data.float() / 255.0 
    test_y = test_loader.dataset.targets
    prediction = predict(model, test_X)
    accuracy = (prediction == test_y).float().mean()
    print(f'Prediction Accuracy: {accuracy:.4f}')

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(test_X[i].numpy(), cmap='gray')  # .numpy() converts tensor to ndarray
        ax.set_title(f'Pred: {prediction[i].item()}')
        ax.axis('off')