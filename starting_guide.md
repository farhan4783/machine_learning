# WebDev LLM Project - Starting Guide

Welcome! This starting guide explains how to spin up both the Machine Learning backend (WebDev LLM) and the Flutter frontend app.

## 1. Prerequisites
- **Python**: 3.8 or newer.
- **Flutter**: Ensure the Flutter SDK is installed and added to your PATH.
- **Compiler/CUDA**: If you want to use GPU, ensure CUDA is available. For CPU, no extra steps are required.

---

## 2. Server/Backend Setup (Python)

The backend handles the WebDev LLM model training and the FastAPI generation server.

### A. Environment Setup
1. Open a terminal and navigate to the project root directory 

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### B. Preprocessing Data
Before training the model, you need to preprocess and format the dataset:
```bash
python data/preprocessor.py
```
This script reads raw data (e.g., from `data/raw/raw_data.json`), augments it into conversational Question/Answer formats, and splits it into `train`, `val`, and `test` data under `data/processed/`.

### C. Training the Model
Next, train the model. Note that the original training script had an autoregressive bug that has now been **fixed**.
```bash
python src/train.py
```
This produces the trained checkpoint `models/checkpoints/best_model.pt` which is required for the API server.

### D. Running the API Server
Once the model is trained, start the FastAPI application:
```bash
cd api
python main.py
```
The API server will run on `http://localhost:8000`. It provides endpoints like `/generate-card` and `/generate-text`.

---

## 3. Frontend Setup (Flutter)

The frontend project provides the interactive user interface.

### A. Launching the App
1. Open a new terminal and navigate to the flutter app folder:
   ```bash
   cd webdev_app
   ```
2. Fetch the required dart packages:
   ```bash
   flutter pub get
   ```
3. Run the application:
   ```bash
   # Run on the web or an emulator (depending on your setup)
   flutter run -d chrome
   ```

### B. Testing
A critical test issue where `widget_test.dart` referenced the wrong app root class has been **fixed**. You can verify that all tests pass by running:
```bash
flutter test
```

## Troubleshooting
- **Missing Checkpoint**: If you try to run `api/main.py` and get "Model not loaded" or a startup warning, ensure you ran the `python src/train.py` script and it successfully created `models/checkpoints/best_model.pt`.
- **Flutter Build Issues**: Run `flutter clean` then `flutter pub get` again to clear cached objects if you encounter strange compile issues.

this is a fear friendly model it takes to much energy and time and needs patience to train
