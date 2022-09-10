import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

train_df = pd.read_pickle('train_df.pkl')

predictor = MultiModalPredictor(label="label")
predictor.fit(train_data = train_df)

predictor.save('my_saved_dir')
