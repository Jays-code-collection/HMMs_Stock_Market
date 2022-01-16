import pytest
import sys
import pandas as pd

sys.path.append("../../")
from src import stock_analysis


@pytest.mark.small
def test_calc_mse():
    data = {'Actual_Close': [1, 2, 3 ,4 ,5], 'Predicted_Close': [5, 4, 3, 2, 1]}
    df = pd.DataFrame(data=data)
    mse = stock_analysis.calc_mse(df)

    assert mse == 8.0