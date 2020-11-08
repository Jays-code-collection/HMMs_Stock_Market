"""
Usage: stock_analysis.py --company=<company>
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
from tqdm import tqdm
from docopt import docopt
import argparse
import sys

# args = docopt(doc=__doc__, argv=None, help=True,
#               version=None, options_first=False)
# Type in python stock_analysis.py --company AAPL, and replace
# AAPL with your desired stock in your command prompt
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")


class StockPredictor:
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
        self.days = 0

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def _split_train_test_data(self, test_size):
        # TODO: Make dates dynamic entries
        # start_date = '2017-01-01'  # change the dates here as desired
        # end_date = '2020-01-01'
        # Use pandas_reader.data.DataReader to load the required financial data. Check if the stock entry is valid.
        try:
            used_data = data.DataReader(self.company, 'yahoo', self.start_date, self.end_date)
        except IOError:
            print("Invalid stock selection. Please try again with a stock that is available on Yahoo finance.")
            sys.exit()
        # Do not shuffle the data as it is a time series
        _train_data, test_data = train_test_split(
            used_data, test_size=test_size, shuffle=False)

        self._train_data = _train_data
        self._test_data = test_data
        self.days = len(test_data)
        # TODO: Remove print statement
        print(self.days)

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
        observations = StockPredictor._extract_features(self._train_data)
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
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = StockPredictor._extract_features(
            previous_data)

        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        #  Get the indices of the most probable outcome and return it
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        #  Predict the close price using the most_probable_outcome and the open price for a given day
        open_price = self._test_data.iloc[day_index]['Open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(
            day_index)
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_period(self, days):
        #  Store all the predicted close prices
        predicted_close_prices = []
        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        return predicted_close_prices

    def actual_close_prices(self):
        #  Store all the actual close prices
        actual_close_prices = self._test_data['Close']
        return actual_close_prices

    # TODO: Add visualisation of results 


# TODO: Add if name = main function that takes in all the required arguments using arg parser
# and add in pathways for whether you want visualisation or not. Required arguments will be:
# output dir, stock name, time period, visualisation y/ n.
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
    args = arg_parser.parse_args()

    # Set variables from arguments
    company_name = args.stock_name
    start = args.start_date
    end = args.end_date

    # Correct incorrect inputs. Inputs should be of the form XXXX, but handle cases when users input 'XXXX'
    if company_name[0] == '\'' and company_name[-1] == '\'':
        company_name = company_name[1:-1]
    elif company_name[0] == '\'' and company_name[-1] != '\'':
        company_name = company_name[1:]
    elif company_name[-1] == '\'' and company_name[0] != '\'':
        company_name = company_name[:-1]


    print(company_name)
    print(start)
    print(end)

    # Initialise StockPredictor object
    stock_predictor = StockPredictor(company=company_name, start_date=start, end_date=end)

    # Fit the HMM
    stock_predictor.fit()

    start_date = '2017-01-01'  # change the dates here as desired
    end_date = '2020-01-01'


print("hi")

# stock_predictor = StockPredictor(company=args['--company'])
# stock_predictor.fit()
# predicted_close = stock_predictor.predict_close_prices_for_period(249, with_plot=False)
# real_close_values = stock_predictor.actual_close_prices()
# print(real_close_values)
# predicted_close_values = pd.DataFrame(predicted_close)
# print(predicted_close_values)
# predicted_close_values.to_excel(r'C:\Users\Sean\Desktop\StudioCodePython\disneypredicted.xlsx', index=False)
# real_close_values.to_excel(r'C:\Users\Sean\Desktop\StudioCodePython\disneyactual.xlsx', index=False)

if __name__ == '__main__':
    # Model prediction scoring is saved in the same directory as the images that are tested.
    main()
