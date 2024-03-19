import os
import pandas as pd
import joblib
from prediction_model.config import config

# Load the dataset
def load_dataset(filename):
    file_path=os.path.join(config.DATA_PATH,filename)
    _data=pd.read_csv(file_path)
    return _data

# Serialization
def save_pipeline(pipeline_to_save):
    save_path=os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved with name {config.MODEL_NAME}")

# DeSerialization
def load_pipeline(pipeline_to_load):
    save_path=os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded=joblib.load(save_path)
    print("Model has been loaded")   
    return model_loaded