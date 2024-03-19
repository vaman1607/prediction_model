from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np, pandas as pd


classification_pipeline=Pipeline(
      [
          ("MeanImputer", pp.MeanImputer(variables=config.NUM_FEATURES)),
          ("ModeImputer", pp.ModeImputer(variables=config.CAT_FEATURES)),
          ("DomainProcessing", pp.DomainProcessing(variables_to_modify=config.FEATURES_TO_MODIFY, variables_to_add=config.FEATURES_TO_ADD)),
          ("DropFeatures", pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
          ("Labelencoder",pp.CustomLabelEncoder(variables=config.CAT_FEATURES)),
          ("LogTranformation", pp.LogTransforms(variables=config.NUM_FEATURES)),
          ("MinMaxScale",MinMaxScaler()),
          ("LogisticClassifier",LogisticRegression(random_state=0))
      ]
)