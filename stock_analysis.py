"""
Usage: stock_analysis.py --n=<stock_name> --s<start_date> --e<end_date> --o<out_directory>
"""
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse
import sys
import os
import time
from datetime import datetime, timedelta

# Supress warning in hmmlearn
warnings.filterwarnings("ignore")


class HMMStockPredictor:
    def __init__(self, company, start_date, end_date, test_size=0.33,
                 n_hidden_states=4, n_latency_days=10,
                 n_intervals_frac_change=50, n_intervals_frac_high=10,
                 n_intervals_frac_low=10):
        self._init_logger()
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.n_latency_days = n_latency_days
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        self._split_train_test_data(test_size)
        self._compute_all_possible_outcomes(
            n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low)
        self.predicted_close = None
        self.future_open = []
        self.future_close = []
        self.future_dates = []
        self.days_in_future = 5

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def _split_train_test_data(self, test_size):
        # Use pandas_reader.data.DataReader to load the required financial data. Check if the stock entry is valid.
        try:
            used_data = data.DataReader(self.company, 'yahoo', self.start_date, self.end_date)
        except IOError:
            print("Invalid stock selection. Please try again with a stock that is available on Yahoo finance.")
            sys.exit()
        # Do not shuffle the data as it is a time series
        _train_data, test_data = train_test_split(
            used_data, test_size=test_size, shuffle=False)
        self.train_data = _train_data
        self.test_data = test_data

        # Drop the columns that aren't used
        self.train_data = self.train_data.drop(['Volume', 'Adj Close'], axis=1)
        self.test_data = self.test_data.drop(['Volume', 'Adj Close'], axis=1)

        # Set days attribute
        self.days = len(test_data)

    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['Open'])
        close_price = np.array(data['Close'])
        high_price = np.array(data['High'])
        low_price = np.array(data['Low'])

        # We compute the fractional change in high,low and close prices
        # to use as our set of observations
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
        # Put the observations into one array
        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        self._logger.info('>>> Extracting Features')
        observations = HMMStockPredictor._extract_features(self.train_data)
        self._logger.info('Features extraction Completed <<<')
        # Fit the HMM using the fit feature of hmmlearn
        self.hmm.fit(observations)

    def _compute_all_possible_outcomes(self, n_intervals_frac_change,
                                       n_intervals_frac_high, n_intervals_frac_low):
        #  Returns np arrays with evenly spaced numbers for each range
        frac_change_range = np.linspace(-0.1, 0.1, n_intervals_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_intervals_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_intervals_frac_low)

        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def _get_most_probable_outcome(self, day_index):
        # Use the previous n_latency_days worth of data for predictions
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self.test_data.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = HMMStockPredictor._extract_features(
            previous_data)

        outcome_score = []
        # Score all possible outcomes and select the most probable one to use for prediction
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        # Get the index of the most probable outcome and return it
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        #  Predict the close price using the most_probable_outcome and the open price for a given day
        open_price = self.test_data.iloc[day_index]['Open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(
            day_index)
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_period(self):
        #  Store all the predicted close prices
        predicted_close_prices = []
        print("Predicting Close prices from " + str(self.test_data.index[0]) + " to " + str(self.test_data.index[-1]))
        for day_index in tqdm(range(self.days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self.predicted_close = predicted_close_prices
        return predicted_close_prices

    def real_close_prices(self):
        #  Store all the actual close prices
        actual_close_prices = self.test_data.loc[:, ['Close']]
        return actual_close_prices

    # TODO: Will require some kind of feeding forward of predictions from the last day onwards. I think it will
    # be easier to append to the original DF, rather than creating an entirely new set of values, based on the
    # current method used for prediction

    def add_future_days(self):
        # Add rows to the test data for the future days being predicted. Leave this empty for now as they will be
        # populated whilst predicting.
        last_day = self.test_data.index[-1] + timedelta(days=self.days_in_future)
        future_dates = pd.date_range(self.test_data.index[-1] + pd.offsets.DateOffset(1), last_day)
        second_df = pd.DataFrame(index=future_dates, columns=['High', 'Low', 'Open', 'Close'])
        final = pd.concat([self.test_data, second_df])
        print(final)

    # add 10 days to day_index, add 10 days to test_data, populate the values in these extra 10 rows as each iteration
    # goes


def plot_results(in_df, out_dir, stock_name):
    """
    Plot the actual and predicted stock prices for a given time period and save the results.
    """
    in_df = in_df.reset_index()  # Required for plotting
    ax = plt.gca()
    in_df.plot(kind='line', x='Date', y='Actual_Close', ax=ax)
    in_df.plot(kind='line', x='Date', y='Predicted_Close', color='red', ax=ax)
    plt.ylabel('Daily Close Price (in USD)')
    plt.title(str(stock_name) + ' daily closing stock prices')
    save_dir = out_dir + '/' + str(stock_name) + 'results_plot' + '.png'
    plt.savefig(save_dir)
    plt.show()
    plt.close('all')


def check_bool(input):
    """
    Corrects an issue that argparser has in which it treats False inputs for a boolean argument as true.
    """
    if isinstance(input, bool):
        return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# TODO: Add in functionality to return most recent stock price prediction so people can use it for
# a given day

# TODO: Add in functionality so that it uses predicted close prices to predict even further into the
# future, maybe 10+ days? So, have it predict the final close price, then make that the open price for
# the next day, and continue! Would need a "days in future" variable too.
def main():
    # Set up arg_parser to handle inputs
    arg_parser = argparse.ArgumentParser()

    # Parse console inputs
    arg_parser.add_argument("-n", "--stock_name", required=True, type=str,
                            help="Takes in the name of a stock in the form XXXX e.g. AAPL. 'AAPL' will fail.")
    arg_parser.add_argument("-s", "--start_date", required=True, type=str,
                            help="Takes in the start date of the time period being evaluated. Please input dates in the"
                                 "following way: 'year-month-day'")
    arg_parser.add_argument("-e", "--end_date", required=True, type=str,
                            help="Takes in the end date of the time period being evaluated. Please input dates in the"
                                 "following way: 'year-month-day'")
    arg_parser.add_argument("-o", "--out_dir", type=str, default=None,
                            help="Directory to save the CSV file that contains the actual stock prices along with the "
                                 "predictions for a given day.")
    arg_parser.add_argument("-p", "--plot", type=check_bool, nargs='?', const=True, default=False,
                            help="Optional: Boolean flag specifying if the results should be plotted or not.")
    args = arg_parser.parse_args()

    # Set variables from arguments
    company_name = args.stock_name
    start = args.start_date
    end = args.end_date
    # Use the current working directory for saving if there is no input
    if args.out_dir is None:
        out_dir = os.getcwd()
    else:
        out_dir = args.out_dir
    plot = args.plot

    # Correct incorrect inputs. Inputs should be of the form XXXX, but handle cases when users input 'XXXX'
    if company_name[0] == '\'' and company_name[-1] == '\'':
        company_name = company_name[1:-1]
    elif company_name[0] == '\'' and company_name[-1] != '\'':
        company_name = company_name[1:]
    elif company_name[-1] == '\'' and company_name[0] != '\'':
        company_name = company_name[:-1]
    print("Using continuous Hidden Markov Models to predict stock prices for " + str(company_name))

    # Initialise HMMStockPredictor object and fit the HMM
    stock_predictor = HMMStockPredictor(company=company_name, start_date=start, end_date=end)
    stock_predictor.fit()
    print("Training data period is from " + str(stock_predictor.train_data.index[0]) + " to " + str(
        stock_predictor.train_data.index[-1]))

    # Get the predicted and actual stock prices and create a DF for saving
    predicted_close = stock_predictor.predict_close_prices_for_period()
    actual_close = stock_predictor.real_close_prices()
    stock_predictor.add_future_days()
    actual_close["Predicted_Close"] = predicted_close
    output_df = actual_close.rename(columns={"Close": "Actual_Close"})

    # Calculate Mean Squared Error and save
    # TODO: put this in a function
    actual_arr = (output_df.loc[:, "Actual_Close"]).values
    pred_arr = (output_df.loc[:, "Predicted_Close"]).values
    mse = mean_squared_error(actual_arr, pred_arr)
    out_name = out_dir + '/' + str(company_name) + '_HMM_Prediction_' + str(round(mse, 6)) + '.xlsx'
    output_df.to_excel(out_name)  # Requires openpyxl installed
    print("All predictions saved. The Mean Squared Error for the " + str(
        stock_predictor.days) + " days considered is: " + str(mse))

    # Plot and save results if plot is True
    if plot:
        plot_results(output_df, out_dir, company_name)

    # python stock_analysis.py -n AAPL -s 2020-01-01 -e 2020-01-15 -o C:\Users\Sean\Desktop\Test


if __name__ == '__main__':
    # Model prediction scoring is saved in the same directory as the images that are tested.
    main()
