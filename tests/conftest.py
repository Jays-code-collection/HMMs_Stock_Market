import pytest 
import sys 
import os 

@pytest.fixture
def company_name():
    yield "AAPL"


def cleanup(file_type):
    files = os.listdir(os.getcwd())
    for file in files:
        if file[-(len(file_type)):] == file_type:
            os.remove(os.path.join(os.getcwd(), file))


@pytest.fixture
def input_args(company_name):
    yield [
            "../../stock_analysis",
            "-n",
            company_name,
            "-s",
            "2020-11-01",
            "-e",
            "2020-12-05",
        ]


@pytest.fixture
def cleanup_excel_files():
    yield 
    cleanup(".xlsx")


@pytest.fixture
def cleanup_images():
    yield
    cleanup(".png")