import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions

# Output from predict script is not null
# Output from predict script is a string
# Output is Y for an example data


# Fixtures--> functions before test function --> ensures single prediction

@pytest.fixture
def single_prediction():
    test_data=load_dataset(config.TEST_FILE)
    single_row=test_data[:1]
    result=generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],str) 

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0]=='Y'
