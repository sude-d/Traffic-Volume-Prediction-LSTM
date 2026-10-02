import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
BATCH_SIZE = 64
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
class GRUNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size
    ):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )
    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        output = self.fc(last_out)
        return output
model = GRUNet(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE
)
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_list = []
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_epoch_loss)
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {avg_epoch_loss:.6f}"
    )
plt.figure()
plt.plot(
    loss_list,
    marker="o"
)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
torch.save(
    model.state_dict(),
    "gru_model.pth"
)
print("Model başarıyla kaydedildi: gru_model.pth")