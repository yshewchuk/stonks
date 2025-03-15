# Stock Price Prediction Project (Stonks)

This project is designed for analyzing stock price data, creating time windows, calculating performance metrics, and training machine learning models to predict future price movements.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setting up the environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stonks.git
cd stonks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### For GPU support (recommended for faster training)
If you have a compatible NVIDIA GPU, you can install the GPU version of TensorFlow:
```bash
pip install tensorflow-gpu
```

## Data Processing Pipeline

The data processing pipeline consists of several steps:

1. Load raw ticker data
2. Isolate training data
3. Calculate training performance metrics 
4. Merge historical data
5. Create training time windows
6. Scale the data

## Training Models

Train price prediction models for specific tickers:

```bash
python -m 12_train_price_prediction_models --tickers AAPL MSFT GOOG
```

For all tickers defined in CONFIG:
```bash
python -m 12_train_price_prediction_models
```

To just check if the required data directories exist:
```bash
python -m 12_train_price_prediction_models --check-only
``` 