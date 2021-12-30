import pandas as pd
from autogluon.vision import ImageDataset
from autogluon.vision import ImagePredictor
from autogluon .core.space import Categorical

train_df = pd.read_pickle('train_df.pkl')

train_dataset = ImageDataset(train_df)

predictor = ImagePredictor()
model = Categorical('resnet18_v1b')
hyperparameters = {'model':model, 'batch_size':4, 'epochs': 3}
hyperparameter_tune_kwargs={'num_trials': 2}
predictor.fit(
    train_dataset, 
    hyperparameters = hyperparameters,
    hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
    )
predictor.save('predictor.ag')