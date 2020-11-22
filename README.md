# Stock Market prediction using Hidden Markov Models
This repo contains all code related to my work using Hidden Markov Models to predict stock market prices. This
initially started as academic work, for my masters dissertation, but has since been a project that I have continued to work on 
post graduation. At present, the program must be called from a terminal/ command line, but there is
an aim to extend it to an interactive site in future, potentially via Django.

## Motivation
Hidden Markov Models are an incredibly interesting type of stochastic process that I believe are under utilised in the
Machine Learning world. They are particularly useful for analysing time series. This, combined with their ability to 
convert the observable outputs that are emitted by real-world processes into predictable and efficient models makes
them a viable candidate to be used for stock market analysis. The stock market
has several interesting properties that make modeling non-trivial, namely
volatility, time dependence and other similar complex dependencies. HMMs
are suited to dealing with these complications as the only information they
require to generate a model is a set of observations (in this case historical stock market data).

## Dependencies
* Pandas_datareader - Allows one to download data directly from Yahoo finance
* NumPy - Required for fast manipulation of financial data (e.g. calculating fractional change)
* Matplotlib - Required for visualisation of results
* Hmmlearn - Open source package that allows for creation and fitting of HMM's 
* Sklearn - Used to calculate metrics to score the results and split the data, will be removed in future to reduce dependency
* Tqdm - Used to track progress whilst training
* Argparse - Required for console inputs

## Method
Stock market data is downloaded via pandas_datareader and the data is split into training and testing datasets. The 
fractional changes for any given day (from open to close, high to open, open to low) in the training dataset are computed and stored in a NumPy 
array. These fractional changes can be seen as the observations for the HMM and are used to train the continuous HMM 
with hmmlearn's fit method. The model then predicts the closing price for each day in the training dataset, based on the given 
days opening price. This prediction is found by calculating the highest scoring potential outcome out of a pre-determined 
set of outcomes (e.g. +0.001%, -0.001% etc). Finally, all predictions as well as the actual close prices for the testing period are stored in an 
excel file and the Mean Squared Error between the two is printed out. The MSE is also included in the file name for future 
reference. 

## Usage 
```shell
python stock_analysis.py [-n XXXX] [-s yyyy-mm-dd] [-e yyyy-mm-dd] [-o dir]
```
The -n input represents a given stock name, -s is the start date of the period considered, -e is the end date of the period considered 
and -o takes in the output directory for the excel file produced. It is important that the dates are input in the correct
order. 

## Example
Input training on and predicting stock prices between January 1st 2020 to February 1st 2020. Typically the model will 
need to be trained on longer periods for more accurate results but this is purely to have a simple example.
Input:
```shell
python stock_analysis.py -n AAPL -s 2020-01-01 -e 2020-02-02 -o C:\Users\Jay\Test
```

Output:
```shell
Using continuous Hidden Markov Models to predict stock prices for AAPL
2020-11-22 15:20:29,810 __main__     INFO     >>> Extracting Features
2020-11-22 15:20:29,811 __main__     INFO     Features extraction Completed <<<
Training data period is from 2020-01-02 00:00:00 to 2020-01-22 00:00:00
Predicting Close prices from 2020-01-23 00:00:00 to 2020-01-31 00:00:00
100%|██████████▍|  7/7 [00:12<00:00,  1.85s/it]
All predictions saved. The Mean Squared Error for the 7 days considered is: 1.9658745300300378
```

Excel file:

|          Date         | Actual_Close | Predicted_Close |
|:---------------------:|:------------:|:---------------:|
| 2020-01-23   00:00:00 | 79.8075      | 79.96662        |
| 2020-01-24   00:00:00 | 79.5775      | 80.55268        |
| 2020-01-27   00:00:00 | 77.2375      | 77.98958        |
| 2020-01-28   00:00:00 | 79.4225      | 78.62847        |
| 2020-01-29   00:00:00 | 81.085       | 81.60911        |
| 2020-01-30   00:00:00 | 80.9675      | 80.62562        |
| 2020-01-31   00:00:00 | 77.3775      | 80.72372        |

