import pandas as pd
from autogluon.text.automm import AutoMMPredictor

test_df = pd.read_pickle('test_df.pkl')

predictor = AutoMMPredictor.load('my_saved_dir')

score = predictor.evaluate(test_df, metrics=["accuracy"])
print(score)
