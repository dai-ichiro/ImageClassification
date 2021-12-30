import pandas as pd
from autogluon.vision import ImageDataset
from autogluon.vision import ImagePredictor

test_df = pd.read_pickle('test_df.pkl')

test_dataset = ImageDataset(test_df)

predictor = ImagePredictor.load('predictor.ag')

result = predictor.evaluate(test_dataset)

print('Top-1 test acc: %.3f' % result['top1'])

