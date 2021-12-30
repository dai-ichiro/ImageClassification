import glob
import random
import pandas as pd

cat_files = glob.glob('train/cat*')
dog_files = glob.glob('train/dog*')

cat_train = random.sample(cat_files, 10000)
dog_train = random.sample(dog_files, 10000)


cat_test = list(set(cat_files) - set(cat_train))
dog_test = list(set(dog_files) - set(dog_train))

train_dataset_list = []
for image_path in cat_train:
    train_dataset_list.append({
        'image': image_path,
        'label': 0
        })
for image_path in dog_train:
    train_dataset_list.append({
        'image': image_path,
        'label': 1
        })
train_df = pd.DataFrame(train_dataset_list)
train_df.to_pickle('train_df.pkl')


test_dataset_list = []
for image_path in cat_test:
    test_dataset_list.append({
        'image': image_path,
        'label': 0
        })
for image_path in dog_test:
    test_dataset_list.append({
        'image': image_path,
        'label': 1
        })
test_df = pd.DataFrame(test_dataset_list)
test_df.to_pickle('test_df.pkl')

