# 🚗 Traffic Volume Prediction – LSTM

## 🇹🇷 Türkçe

LSTM tabanlı bir derin öğrenme modeli kullanılarak **24 saatlik trafik hacmi tahmini** yapan bir zaman serisi projesidir.

Projede hava durumu ve zaman bilgileri kullanılarak trafik hacminin tahmin edilmesi amaçlanmıştır. Model, FastAPI üzerinden servis edilmekte ve Streamlit ile kullanıcı dostu bir arayüz üzerinden kullanılabilmektedir.

## 🇬🇧 English

This is a **time-series traffic volume prediction project** using an LSTM-based deep learning model to predict traffic volume for the next 24 hours.

The project uses weather and time-related features to predict traffic volume. The trained model is served through FastAPI and can be used through a user-friendly Streamlit interface.

---

# 📌 Proje Özellikleri | Project Features

### 🇹🇷 Türkçe

- 24 saatlik veri girişi
- Sıcaklık, yağmur, kar ve bulutluluk bilgileri
- Saat, haftanın günü ve ay bilgileri
- LSTM tabanlı trafik tahmini
- FastAPI ile REST API
- Streamlit ile web arayüzü
- JSON tabanlı API iletişimi

### 🇬🇧 English

- 24-hour input sequence
- Temperature, rain, snow and cloud information
- Hour, day of week and month features
- LSTM-based traffic prediction
- REST API with FastAPI
- Web interface with Streamlit
- JSON-based API communication

---

# 🛠️ Kullanılan Teknolojiler | Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Streamlit
- Requests

---

# 📊 Kullanılan Özellikler | Input Features

| Özellik | Açıklama | Feature | Description |
|---|---|---|---|
| Temperature | Sıcaklık (Kelvin) | Temperature | Temperature in Kelvin |
| Rain | Yağış miktarı (mm) | Rain | Rainfall in mm |
| Snow | Kar miktarı (mm) | Snow | Snowfall in mm |
| Clouds | Bulutluluk (%) | Clouds | Cloud coverage (%) |
| Hour | Saat (0–23) | Hour | Hour of the day |
| Day of Week | Haftanın günü (0–6) | Day of Week | Day of the week |
| Month | Ay (1–12) | Month | Month of the year |

---

# 🧠 Model | Model

### 🇹🇷 Türkçe

Projede trafik hacmi tahmini için **LSTM (Long Short-Term Memory)** modeli kullanılmıştır.

Model, geçmiş zaman adımlarındaki bilgileri kullanarak trafik hacmindeki zaman bağımlılıklarını öğrenir.

### 🇬🇧 English

The project uses an **LSTM (Long Short-Term Memory)** model for traffic volume prediction.

The model learns temporal dependencies in traffic volume by using information from previous time steps.

---

# 🔌 API

### 🇹🇷 Türkçe

Model, FastAPI kullanılarak bir REST API olarak sunulmaktadır.

API adresi:

```text
http://127.0.0.1:8000
```

Tahmin endpoint'i:

```text
POST /predict
```

Örnek istek:

```json
{
    "sequence": [
        [290.0, 0.0, 0.0, 50, 0, 0, 1],
        [291.0, 0.0, 0.0, 45, 1, 0, 1]
    ]
}
```

Gerçek kullanımda `sequence` içerisinde 24 saatlik veri gönderilir.

### 🇬🇧 English

The model is served as a REST API using FastAPI.

API address:

```text
http://127.0.0.1:8000
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```json
{
    "sequence": [
        [290.0, 0.0, 0.0, 50, 0, 0, 1],
        [291.0, 0.0, 0.0, 45, 1, 0, 1]
    ]
}
```

In actual usage, the `sequence` contains 24 hours of input data.

---

# 🖥️ Streamlit Arayüzü | Streamlit Interface

### 🇹🇷 Türkçe

Streamlit arayüzü üzerinden kullanıcı her saat için gerekli parametreleri girebilir ve tahmin sonucunu görüntüleyebilir.

### 🇬🇧 English

Through the Streamlit interface, users can enter the required parameters for each hour and view the prediction result.

---

# ▶️ Kurulum | Installation

Projeyi klonladıktan sonra gerekli kütüphaneleri yükleyin.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Gerekli paketleri yükleyin:

```bash
pip install numpy pandas scikit-learn tensorflow fastapi uvicorn streamlit requests
```

---

# 🚀 Çalıştırma | Running the Project

## 1. FastAPI'yi çalıştırın | Run FastAPI

```bash
uvicorn dierpro_main_api:app --reload
```

API şu adreste çalışacaktır:

```text
http://127.0.0.1:8000
```

## 2. Streamlit'i çalıştırın | Run Streamlit

Yeni bir terminal açın ve:

```bash
streamlit run app.py
```

Ardından tarayıcıdan:

```text
http://localhost:8501
```

adresine gidin.

---

# 📁 Proje Yapısı | Project Structure

```text
Traffic-Volume-Prediction/
│
├── dierpro_main_api.py
├── app.py
├── model/
│   └── traffic_model.keras
│
├── scaler.save
├── requirements.txt
└── README.md
```

---

# 🔄 Proje Akışı | Project Workflow

```text
User Input
    ↓
Streamlit
    ↓
FastAPI
    ↓
Data Preprocessing
    ↓
LSTM Model
    ↓
Traffic Volume Prediction
    ↓
Streamlit Result
```

---

# 🎯 Amaç | Purpose

### 🇹🇷 Türkçe

Bu projenin amacı, zaman serisi verileri üzerinde LSTM kullanarak trafik hacmi tahmini gerçekleştirmek ve eğitilmiş bir makine öğrenmesi modelini API ve web arayüzü üzerinden kullanılabilir hale getirmektir.

### 🇬🇧 English

The purpose of this project is to perform traffic volume prediction using LSTM on time-series data and make the trained machine learning model accessible through an API and web interface.

---

# 🔮 Gelecekte Yapılabilecek Geliştirmeler | Future Improvements

### 🇹🇷 Türkçe

- Daha büyük ve güncel veri setleri kullanmak
- Model performansını farklı metriklerle değerlendirmek
- Daha uzun zaman aralıkları için tahmin yapmak
- Gerçek zamanlı trafik verilerini kullanmak
- Model mimarisini geliştirmek
- Grafiksel tahmin sonuçları eklemek

### 🇬🇧 English

- Use larger and more recent datasets
- Evaluate model performance with additional metrics
- Perform predictions for longer time periods
- Integrate real-time traffic data
- Improve the model architecture
- Add graphical prediction results

---

# 👩‍💻 Author

**Sude Dağaşan**

Python • Machine Learning • Deep Learning • Time Series
