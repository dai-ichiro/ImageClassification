import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor.load('my_saved_dir')

test_pic = "test1.jpg"
proba = predictor.predict_proba(pd.DataFrame({'image':[test_pic]}))
print(proba)