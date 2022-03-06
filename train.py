import os
import pandas as pd
from autogluon.text.automm import AutoMMPredictor

train_df = pd.read_pickle('train_df.pkl')

train_df['image'] = train_df['image'].apply(lambda x: os.path.abspath(x))

predictor = AutoMMPredictor(label='label')
predictor.fit(
    train_data=train_df,
    hyperparameters={
        "model.names": ["timm_image"],
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
    }
)

predictor.save('my_saved_dir')


