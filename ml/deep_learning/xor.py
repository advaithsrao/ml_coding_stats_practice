import torch
from torch import nn
from torch.optim import Adam


class XORPredictor(nn.Module):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.n_examples, input_dim = X.shape
        
        hidden_dim = 2  # Slightly higher capacity
        output_dim = y.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        self.loss = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.1)
    
    def forward(self, X):
        return self.model(X)
    
    def train(self, num_epochs=10000):
        prev_loss = torch.inf
        for epoch in range(num_epochs):
            prediction = self.forward(self.X)
            loss = self.loss(prediction, self.y)
            if epoch % 1000 == 0:
                print(f'Loss at epoch {epoch}: {loss.item()}')
            if torch.abs(prev_loss - loss) < 1e-10:
                print(f'Early stopping at epoch {epoch} due to minimal loss change')
                break
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            prev_loss = loss
    
    def predict(self, X):
        with torch.no_grad():
            prediction = self.model(X)
        return torch.round(prediction)


if __name__ == "__main__":
    # XOR gate
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0 ,1, 1, 0]], dtype=torch.float32).view(-1, 1)
    model = XORPredictor(X, y)
    model.train()
    print('Training Done')
    out = model.predict(X)
    print(f'{out = }')
    print(list(model.parameters()))
