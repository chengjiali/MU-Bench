import pandas as pd

train = pd.read_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/trainlist01.txt', sep=' ', names=['path', 'label'])
test = pd.read_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/testlist01.txt', names=['path'])
label_map = pd.read_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/classInd.txt', sep=' ', names=['label_id', 'label_name'])
label_map = {i: j-1 for i, j in zip(label_map.label_name, label_map.label_id)}

train['label'] = [label_map[i.split('/')[0]] for i in train.path]
test['label'] = [label_map[i.split('/')[0]] for i in test.path]

train.to_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/train.txt', sep=' ', index=None, header=None)
test.to_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/test.txt', sep=' ', index=None, header=None)