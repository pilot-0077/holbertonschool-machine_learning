![Bitcoin forecasting cover](bitcoin_forecasting_cover.svg)

# Can We Forecast Bitcoin Prices with Deep Learning?

*My journey from raw market data to next-hour predictions using TensorFlow and GRU networks.*

> **Draft note:** This article is not ready to publish until the real experiment has been run, the result placeholders below have been replaced, and the generated graphs have been inserted.

Bitcoin prices can move quickly. News, regulation, interest rates, liquidity, and investor behaviour can all influence the market. That makes Bitcoin an interesting challenge for machine learning — and a good reminder that a model's prediction is never a promise.

In this project, I built a time-series forecasting pipeline to explore a focused question: can the previous 24 hours of Bitcoin market data help estimate the next hour's closing price?

The project uses Python, pandas, NumPy, TensorFlow, and a recurrent neural network called a GRU. This article explains the process in plain language so that another learner can reproduce the experiment.

## 1. What is time-series forecasting?

A time series is a sequence of observations collected over time. Hourly Bitcoin closing prices are one example. Electricity demand, daily sales, and weather measurements are others.

Time-series forecasting uses earlier observations to estimate a future value. The order of the data matters: when predicting a future price, we must not let information from that future period leak into training.

My model uses 24 consecutive hourly observations as input and predicts one value: the next hourly closing price.

For example:

```text
Input:  Hour 1, Hour 2, ... Hour 24
Target: Hour 25 closing price

Input:  Hour 2, Hour 3, ... Hour 25
Target: Hour 26 closing price
```

This moving input window is called a sliding window.

## 2. Preprocessing: making raw data useful

The market dataset contains minute-level Bitcoin observations. I aggregate them into hourly rows because the task is to estimate an hourly close.

For each hour, the script calculates the first open, highest high, lowest low, last close, and total BTC and currency volumes. If the source file contains only BTC volume, currency volume is estimated from BTC volume multiplied by the close; this is an approximation, not an exchange-reported USD volume.

Real datasets also contain missing intervals. The preprocessing code fills missing closing prices forward, fills missing open/high/low values with the closing price, and sets missing volume values to zero. These choices keep the series usable, although filled values should always be treated cautiously.

Next, I standardize the features:

```text
standardized value = (value - training mean) / training standard deviation
```

The mean and standard deviation are calculated from the training period only. This is important: using statistics from the future validation or test periods would leak information into the model.

I split the observations chronologically:

- First 70% — training
- Next 15% — validation, used to monitor training and select the best model
- Final 15% — held-out test set, used for final evaluation

This is a more realistic test than randomly mixing past and future observations.

## 3. Building the TensorFlow dataset

TensorFlow's `timeseries_dataset_from_array` creates the 24-hour input windows and pairs them with their next-hour targets.

```python
dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=window_data,
    targets=labels,
    sequence_length=24,
    sequence_stride=1,
    sampling_rate=1,
    shuffle=shuffle,
    batch_size=64,
)
```

Here, `sequence_length=24` means that each example contains 24 hourly observations. `sequence_stride=1` moves the window forward by one hour. `batch_size=64` lets the model process up to 64 examples at a time.

I also use prefetching to prepare upcoming batches while the model works on the current batch.

The training windows can be shuffled, but the validation and test windows remain chronological. The model never trains on the held-out test targets.

## 4. Why a GRU model?

A recurrent neural network is designed to process sequences. A GRU — or Gated Recurrent Unit — uses gates to help retain or update information as it moves through the sequence.

The model architecture is:

1. GRU layer with 64 units, returning its full sequence
2. GRU layer with 32 units
3. Dense layer with 16 neurons and ReLU activation
4. Dense output layer with one predicted value

I use the Adam optimizer and mean squared error (MSE) as the loss function. I also track mean absolute error (MAE). Early stopping monitors validation loss, and model checkpointing saves the best model.

## 5. Results: does the model beat a simple baseline?

A model that trains successfully is not necessarily useful. To make the evaluation meaningful, I compare the GRU against a simple baseline: predict that the next closing price will equal the most recent closing price.

**Replace the values below with the measured output from the actual run. Do not publish this section with placeholders.**

| Test metric | Market-only GRU | Previous-close baseline |
|---|---:|---:|
| MAE in USD | [insert result] | [insert result] |
| RMSE in USD | [insert result] | [insert result] |

Also report the normalized test MSE and MAE from the training script. Lower error is better, but the baseline comparison matters: if the GRU cannot beat a simple previous-close prediction, the more complex model may not add practical value.

### Graph 1: Training and validation loss

Insert `results/market_only/training_validation_loss.png` after running the experiment. This graph helps show whether validation performance improves along with training performance or whether the model begins to overfit.

### Graph 2: Actual versus predicted closing prices

Insert `results/market_only/actual_vs_predicted_test.png`. This graph compares real test-period closing prices with the GRU predictions and the previous-close baseline.

### Optional extension: adding news sentiment

Market prices alone do not capture every event. The repository therefore supports an optional experiment that adds hourly news sentiment and article counts.

The news-enhanced model should be trained and tested over the same dates and with the same chronological split as the market-only model. Only then is the comparison fair.

If the news-enhanced model achieves lower held-out test error, that is evidence that the added features helped on this test period — not proof that news will always improve forecasts.

## 6. What I learned

This project taught me that forecasting is not just about selecting a neural-network architecture. Data quality, chronological splitting, feature scaling, input-window alignment, and a meaningful baseline are just as important.

It also helped me separate an interesting model from a useful one. A graph may look convincing, but measured test performance is what matters.

Bitcoin remains difficult to forecast because historical prices and sentiment cannot capture every surprise. News can be misinterpreted, a market can change regime, and sudden events can overwhelm patterns learned from the past.

## Conclusion

Building this pipeline gave me practical experience with preprocessing real-world data, creating TensorFlow datasets, training a GRU network, and evaluating forecasts on data held out from model selection.

The goal is not to claim that deep learning can predict Bitcoin perfectly. It is to build a reproducible experiment, measure its limitations, and improve it only when the evidence supports the change.

**Code:** [GitHub — Time Series Forecasting](https://github.com/pilot-0077/holbertonschool-machine_learning/tree/main/supervised_learning/time_series)

*This project is for educational purposes only and is not financial advice.*
