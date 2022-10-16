import os
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

seed = 0
def write_ann_file(file, ids):
    with open(file, 'w') as f:
        for id in ids:
            f.write(f"{id}\n")

def main(proportion, train_size):
    if train_size and train_size <= 0:
            raise ValueError("Train size must be positive")

    if proportion and (proportion <= 0.0 or proportion >= 1.0):
            raise ValueError("Proportion must be float between 0 and 1")

    train_data = pd.read_csv('data/annotations.csv')

    train_data['source']= pd.factorize(train_data['source'])[0]
    train_data['source'] = train_data['source'] + 1

    imgs_regions = train_data.groupby('image_id')['source'].first()
    img_ids = imgs_regions.index.to_numpy(dtype=str)
    regions = imgs_regions.to_numpy(dtype=np.int64)

    non_labeled = 0
    for f in os.listdir('data/train'):
        id = os.path.splitext(os.path.basename(f))[0]
        if id not in img_ids:
            non_labeled += 1
            np.append(img_ids, id)
            np.append(regions, 0)

    print(f'{non_labeled} unlabeled images were found')

    if train_size:
        partial_size = min(int(train_size / proportion), len(img_ids))
        img_ids = img_ids[:partial_size]
        regions = regions[:partial_size]

    train_ids, val_ids = train_test_split(img_ids, train_size=proportion, stratify=regions, random_state=seed)
    test_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir('data/test')]

    write_ann_file('data/train.txt', train_ids)
    write_ann_file('data/val.txt', val_ids)
    write_ann_file('data/test.txt', test_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proportion', type=float, default=0.85,
                        help='proportion of the dataset to include in the train split')
    parser.add_argument('--train_size', type=int, default=None,
                        help='number of images in train split')
    opt = parser.parse_args()
    main(opt.proportion, opt.train_size)