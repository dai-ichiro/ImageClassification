import pandas as pd
from autogluon.text.automm import AutoMMPredictor

predictor = AutoMMPredictor.load('my_saved_dir')

test_pic = "test1.jpg"
proba = predictor.predict_proba(pd.DataFrame({'image':[test_pic]}))
print(proba)

