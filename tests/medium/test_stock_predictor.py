import pytest
import sys

print(sys.path)
sys.path.append("../../")
from src import stock_analysis


@pytest.mark.medium
def test_create_stock_predictor_valid_dates(
    company_name, valid_start_date, valid_end_date
):
    # Given
    stock_predictor = stock_analysis.HMMStockPredictor(
        company=company_name,
        start_date=valid_start_date,
        end_date=valid_end_date,
        future_days=0,
    )
    assert stock_predictor.start_date == valid_start_date
    assert stock_predictor.end_date == valid_end_date
    assert stock_predictor.company == company_name


@pytest.mark.medium
def test_create_stock_predictor_invalid_dates(
    company_name, invalid_start_date, valid_end_date
):
    # Given
    with pytest.raises(ValueError):
        stock_predictor = stock_analysis.HMMStockPredictor(
            company=company_name,
            start_date=invalid_start_date,
            end_date=valid_end_date,
            future_days=0,
        )