import os.path as osp
import os  
import random  
import argparse
from glob import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='/project/train/dataset', type=str, help='input xml label path')
parser.add_argument('--root', default='/project/train/dataset', type=str, help='output txt label path')
args = parser.parse_args()

root = args.root
save_path = args.save_path

imgpath = glob(osp.join(root, 'images') + '/*')

train_data, val_data = train_test_split(imgpath, test_size=0.2, shuffle=True, random_state=233)

print("train data: ", len(train_data))
print("val data: ", len(val_data))
print(len(val_data) / len(train_data))
print("Total:", len(train_data) + len(val_data))

train_txt_path = os.path.join(root, "train.txt")
val_txt_path = os.path.join(root, "val.txt")

train_txt = open(train_txt_path, "w")
val_txt = open(val_txt_path, "w")

random.shuffle(train_data)
random.shuffle(val_data)

for data in train_data:
    train_txt.writelines(data + "\n")

for data in val_data:
    val_txt.writelines(data + "\n")

train_txt.close()
val_txt.close()
    