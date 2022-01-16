import pytest
import os
import sys
from unittest.mock import patch

sys.path.append("../../")
from src import stock_analysis


@pytest.mark.large
def test_test():
    assert True


@pytest.mark.large
def test_stock_analysis_record_metrics(company_name, input_args, cleanup_excel_files):
    input_args.append("-m")
    input_args.append("True")
    # Given, when
    with patch(
        "sys.argv",
        input_args,
    ):
        stock_analysis.main()
    files = os.listdir(os.getcwd())
    # Then
    historical_metrics = [file for file in files if file[-5:] == ".xlsx"]
    assert 1 == len(historical_metrics)
    assert company_name in historical_metrics[0]


@pytest.mark.large
def test_stock_analysis_future_predictions(
    company_name, input_args, cleanup_excel_files
):
    input_args.append("-f")
    input_args.append("5")
    # Given, when
    with patch(
        "sys.argv",
        input_args,
    ):
        stock_analysis.main()
    files = os.listdir(os.getcwd())
    # Then
    predictions = [file for file in files if file[-5:] == ".xlsx"]
    assert 1 == len(predictions)
    assert company_name in predictions[0]


@pytest.mark.large
def test_stock_analysis_plot_image(
    company_name, input_args, cleanup_images, cleanup_excel_files
):
    input_args.append("-m")
    input_args.append("True")
    input_args.append("-p")
    input_args.append("True")
    # Given, when
    with patch(
        "sys.argv",
        input_args,
    ):
        stock_analysis.main()
    files = os.listdir(os.getcwd())
    # Then
    plots = [file for file in files if file[-4:] == ".png"]
    assert 1 == len(plots)
    assert company_name in plots[0]