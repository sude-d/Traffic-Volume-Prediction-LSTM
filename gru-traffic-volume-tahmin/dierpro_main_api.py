from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn
import random
import joblib
app = FastAPI(title="Traffic Volume Prediction API")
class InputData(BaseModel):
    sequence: list
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
model = GRUNet(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE
)
model.load_state_dict(
    torch.load("gru_model.pth", map_location=torch.device("cpu"))
)
model.eval()
scaler_y = joblib.load("scaler_y.save")
@app.post("/predict")
def predict(input_data: InputData):
    input_seq = np.array(
        input_data.sequence,
        dtype=np.float32
    )
    if input_seq.shape != (24, 7):
        return {
            "error": "Input sequence must have shape (24, 7)"
        }
    input_tensor = torch.tensor(
        input_seq,
        dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.numpy()
    prediction_orig = scaler_y.inverse_transform(prediction)
    predicted_value = float(
        prediction_orig[0][0]
    )
    return {
        "predicted_traffic_volume": predicted_value
    }
sample_sequence = []
for i in range(24):
    temp = random.uniform(280, 300)
    rain = 0
    snow = 0
    clouds = random.randint(0, 100)
    hour = i
    day_of_week = random.randint(0, 6)
    month = random.randint(1, 12)
    sample_sequence.append([
        temp,
        rain,
        snow,
        clouds,
        hour,
        day_of_week,
        month
    ])
print(np.array(sample_sequence).shape)
print(sample_sequence)