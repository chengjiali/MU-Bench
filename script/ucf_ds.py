import pandas as pd

ds = pd.read_csv(f'./data/ucf101-DS_eval.csv')
print(ds.label.unique().shape)
label_map = pd.read_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/classInd.txt', sep=' ', names=['label_id', 'label_name'])
label_map = {i: j-1 for i, j in zip(label_map.label_name, label_map.label_id)}
label_map['BilliardsShot'] = label_map['Billiards']
label_map['BasketballShooting'] = label_map['Basketball']
label_map['CleanandJerk'] = label_map['CleanAndJerk']
label_map['Breaststroke'] = label_map['BreastStroke']


ds['path'] = [i + '/' + j for i, j in zip(ds.label, ds.filename)]
ds['label'] = [label_map[i] for i in ds.label]

ds[['path', 'label']].to_csv('./data/ucf101_ds.txt', sep=' ', index=None, header=None)