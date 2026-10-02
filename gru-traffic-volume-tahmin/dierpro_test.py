import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out
SEQ_LEN = 24
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load("gru_model.pth"))
model.eval()
with torch.no_grad():
    predictions =model(X_test_tensor)
predictions = predictions.numpy()
scaler_y = joblib.load("scaler_y.save")
predictions_orig=scaler_y.inverse_transform(predictions)
y_test_orig=scaler_y.inverse_transform(y_test)
rmse = np.sqrt(root_mean_squared_error(y_test_orig, predictions_orig))
mae = mean_absolute_error(y_test_orig, predictions_orig)    
plt.figure()
plt.plot(y_test_orig[:200], label="Actual Traffic Volume", color="steelblue")
plt.plot(predictions_orig[:200], label="Predicted Traffic Volume", color="orange")
plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Time Steps")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()