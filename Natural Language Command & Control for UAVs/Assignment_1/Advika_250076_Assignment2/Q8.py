import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def train_step(model, inputs, targets, optimizer, criterion):

    predictions = model(inputs)
    
    loss = criterion(predictions, targets)
    
    optimizer.zero_grad()    
    loss.backward()          
    optimizer.step()         
    
    return loss.item()

# Test

model = SimpleClassifier()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

dummy_input = torch.rand(1, 10)  # Batch size 1, 10 features
dummy_target = torch.tensor([[1.0]]) # Target label


loss_val = train_step(model, dummy_input, dummy_target, optimizer, criterion)
print(f"Training step complete. Loss: {loss_val:.4f}")