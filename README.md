# Stock Price Prediction Using LSTM and ARIMA Models

This project aims to compare LSTM (Long Short-Term Memory) and ARIMA (AutoRegressive Integrated Moving Average) models to predict stock prices using the Yahoo Finance dataset. The project utilizes data preprocessing, model building, hyperparameter tuning, and performance evaluation to analyze the effectiveness of both models.

## Project Overview

Stock price prediction is a critical problem in the financial world. In this project, we compare two popular approaches for time series forecasting:

1. **LSTM Model** - A deep learning model capable of capturing complex temporal dependencies in sequential data.
2. **ARIMA Model** - A traditional statistical method widely used for time series forecasting.

The objective is to determine which model provides better accuracy and correlation for predicting stock prices.

## Dataset

The project uses stock price data from Yahoo Finance. The dataset includes historical stock prices with the following main feature:

- **Close Price**: The final trading price of the stock for each day.

## Project Goals

1. Fetch historical stock price data from Yahoo Finance.
2. Preprocess the data for training and testing LSTM and ARIMA models.
3. Train and evaluate both models (LSTM and ARIMA).
4. Compare the performance metrics of these models, such as RMSE, MAE, and R² score.
5. Visualize the predictions and true prices for comparison.

## Setup Instructions

1. **Clone the Repository**

   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/ahmdmohamedd/stock_price_prediction_lstm_arima.git
   ```

2. **Install Required Libraries**

   Ensure that you have the required libraries installed. You can install them using:

   ```bash
   pip install numpy pandas matplotlib yfinance statsmodels tensorflow scikit-learn pmdarima
   ```

3. **Launch Jupyter Notebook**

   Run the following command to start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. **Open the Notebook**

   Open the file `stock_price_prediction.ipynb` in Jupyter Notebook to explore the project step-by-step.

## Code Overview

### Data Acquisition and Preprocessing

- The project fetches stock price data from Yahoo Finance using the `yfinance` library.
- The dataset is normalized using `MinMaxScaler` to improve LSTM training efficiency.
- A train-test split is applied to evaluate model predictions accurately.

### Model Training

- **LSTM Model**: Built using TensorFlow's Keras API, which captures temporal patterns in stock prices.
- **ARIMA Model**: A statistical approach implemented with the `statsmodels` library, ensuring comparison with traditional forecasting methods.

### Hyperparameter Tuning

- The LSTM model is trained with various hyperparameters like **batch size**, **number of epochs**, and **number of LSTM units**.
- The ARIMA model optimizes its order parameters (p, d, q) using manual tuning or automatic search methods like `pmdarima`.

### Evaluation Metrics

- The models are evaluated using multiple performance metrics:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² Score (Coefficient of Determination)
  
- Visual plots are generated to compare true prices against model predictions.

## Project Outputs

- Visual comparisons of LSTM and ARIMA predictions.
- Performance metrics to evaluate and contrast the accuracy and reliability of each model.
- Insights into stock price trends and prediction correlations.

## Potential Improvements

- **Better Hyperparameter Tuning**: Use AutoARIMA or Grid Search to optimize ARIMA parameters.
- **Feature Engineering**: Include more financial indicators (SMA, EMA, MACD).
- **Advanced Models**: Explore Transformer-based models or Bidirectional LSTM layers.
- **Data Integration**: Combine features like stock volume and other financial indicators.

## Contribution

If you want to contribute to this project, please fork the repository, make your changes, and create a pull request. Contributions related to model enhancements, optimization, and data preprocessing are welcome.
