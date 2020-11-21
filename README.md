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
days opening price. Finally, all predictions as well as the actual close prices for the testing period are stored in an 
excel file and the Mean Squared Error between the two is printed out. The MSE is also included in the file name for future 
reference. 


