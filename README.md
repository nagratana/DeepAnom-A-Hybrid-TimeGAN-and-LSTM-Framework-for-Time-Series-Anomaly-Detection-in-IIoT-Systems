# DeepAnom: A Hybrid TimeGAN and LSTM Framework for Time Series Anomaly Detection in IIoT Systems

## Objective

This project leverages an IIoT Edge Computing dataset to:

- Learn normal machine-sensor behavior from time-series data
- Generate realistic synthetic normal sequences using TimeGAN
- Train an LSTM Autoencoder anomaly detector to identify failures

## Dataset

- Source: IIoT Edge Computing dataset (included in the `data/` directory)
- Contains timestamped sensor readings: Temperature, Pressure, Vibration, Network Latency, Edge Processing Time, Fuzzy PID Output, and Predicted Failure labels.

## Project Structure

- `app.py`: Flask web application providing the API and serving the UI.
- `model/final-project.ipynb`: Main Jupyter notebook containing data exploration, model training, TimeGAN synthesis, and Autoencoder validation.
- `data/`: Contains the original IIoT dataset and sample test data for predictions.
- `model/`: Stores the trained LSTM autoencoder metrics and Scikit-learn scaler.
- `templates/`: Contains HTML and unified UI components for the web dashboard.

## Workflow

1. Data Loading & Cleaning: Load the dataset, handle missing values, convert timestamps, and preprocess features.
2. Exploratory Data Analysis (EDA): Visualize feature distributions, correlations, and time-series trends.
3. Windowing: Transform data into sequences for time-series modeling (24 timesteps per window).
4. Synthetic Data Generation (TimeGAN): Train a simplified TimeGAN to generate realistic normal sequences to improve detection boundaries.
5. Anomaly Detection (LSTM Autoencoder): Train the autoencoder to detect failures based on reconstruction error (MSE thresholding).
6. Web Deployment: Interface the model with a Flask web application, allowing batch uploads or quick predictions.

## Running the Application Locally

1. Create a virtual environment and install the dependencies.
2. Run the command `python app.py`.
3. The dashboard will be accessible via browser at `http://localhost:81/`.

## Author
Provided by the user under week 6 major project.
