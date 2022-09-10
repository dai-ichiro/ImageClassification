import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

test_df = pd.read_pickle('test_df.pkl')

predictor = MultiModalPredictor.load('my_saved_dir')

score = predictor.evaluate(test_df, metrics=["accuracy"])
print(score)