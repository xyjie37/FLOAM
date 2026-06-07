import os.path
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from data_util import split_data, save_file

num_client = 20
num_task = 5  # 20 classes per task (200 total)
num_classes = 200
alpha = 0.1
np.random.seed(2266)

# Dataset storage path
datasetroot_dir = "/root/tiny-imagenet-200"
# Output dataset path
basedir = "./tinyimagenet-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Tiny-ImageNet normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                         std=[0.2302, 0.2265, 0.2262]),
])

class TinyImageNetDataset:
    def __init__(self, root_dir, mode='train'):
        self.images = []
        self.labels = []
        
        if mode == 'train':
            path = os.path.join(root_dir, 'train')
            classes = sorted(os.listdir(path))
            self.class_to_idx = {cls:i for i, cls in enumerate(classes)}
            for cls in classes:
                cls_path = os.path.join(path, cls, 'images')
                for img in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls])
        else:
            path = os.path.join(root_dir, 'val')
            with open(os.path.join(path, 'val_annotations.txt')) as f:
                lines = f.readlines()
            
            img_to_cls = {line.split('\t')[0]: line.split('\t')[1] 
                         for line in lines}
            classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
            self.class_to_idx = {cls:i for i, cls in enumerate(classes)}
            
            for img in os.listdir(os.path.join(path, 'images')):
                if img in img_to_cls:
                    cls = img_to_cls[img]
                    img_path = os.path.join(path, 'images', img)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

# Load dataset
print("Loading dataset...")
train_dataset = TinyImageNetDataset(datasetroot_dir, 'train')
test_dataset = TinyImageNetDataset(datasetroot_dir, 'val')

total_image_paths = train_dataset.images + test_dataset.images
total_label = np.array(train_dataset.labels + test_dataset.labels)

# Load and transform images
print("Loading and transforming images...")
total_image = []
for img_path in total_image_paths:
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    total_image.append(img.numpy())
total_image = np.array(total_image)

image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]

# Index mapping: client_id -> list of indices
dataidx_map = {}
# All sample indices
idxs = np.array(range(len(total_label)))
# Per-class sample indices
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_label == i])

# Process each class
for i in range(num_classes):
    num_images = len(idx_for_each_class[i])
    num_per_client = num_images / num_client
    per_client_image_number = [int(num_per_client) for _ in range(num_client)]
    idx = 0
    for client, num_sample in enumerate(per_client_image_number):
        if client not in dataidx_map.keys():
            dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
        else:
            dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
        idx += num_sample

# Collect per-client indices
df = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_image[idxs]
    label_per_client[client] = total_label[idxs]
    row = [0 for i in range(num_classes)]
    for i in np.unique(label_per_client[client]):
        statistic[client].append((int(i), int(sum(label_per_client[client] == i))))
        row[i] = int(sum(label_per_client[client] == i))
    df.loc[len(df)] = row
df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    print(f"Client {client}\t Size of data: {len(image_per_client[client])}\t Labels: ", np.unique(label_per_client[client]))
    print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    print("=" * 50)

K = num_classes
least_samples = len(image_per_client[0]) // 10
if num_task == 10:
    least_samples = len(image_per_client[0]) // 20
print("least samples:", least_samples)
N = least_samples

col = [str(i) for i in range(num_classes)]
col.append('client')
col.append('task')
df = pd.DataFrame(columns=col)

# Split each class across tasks via Dirichlet
for client_id in range(num_client):
    client_images = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    # Class-incremental mode - 20 classes per task
    classes_per_task = num_classes // num_task  # 200/5 = 40 classes per task
    for task in range(num_task):
        #start_class = task * classes_per_task
        #end_class = (task + 1) * classes_per_task
        start_class = 0
        end_class = 199
        task_classes = list(range(start_class, end_class + 1))

        min_size = 0
        max_attempts = 1000  # Max retry attempts
        attempt = 0

        
        idx_batch = [[] for _ in range(num_task)]
        for k in task_classes:
            idx_k = np.where(client_dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_task))
            proportions = np.clip(proportions, a_min=0.05, a_max=None)  # Enforce minimum proportion
            proportions /= proportions.sum()  # Renormalize
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_task):
            client_idx_map[j] = idx_batch[j]

    for task_id in range(num_task):
        row = [0 for i in range(num_classes + 2)]
        row[num_classes] = client_id
        row[num_classes + 1] = task_id
        idxs = client_idx_map[task_id]
        Y[task_id] = client_dataset_label[idxs]
        X[task_id] = client_images[idxs]

        info = []
        for i in np.unique(Y[task_id]):
            info.append((int(i), int(sum(Y[task_id] == i))))
            row[i] = int(sum(Y[task_id] == i))
        df.loc[len(df)] = row

        print(f"Client {client_id}  Task {task_id}\t Size of data: {len(X[task_id])}\t Labels: ", np.unique(Y[task_id]))
        print(f"\t\t Samples of labels: ", [i for i in info])
        print("-" * 50)
        print("=" * 50 + "\n\n")

    # Save split data
    train_data, test_data = split_data(X, Y)

    if not os.path.exists(basedir + "/train"):
        os.mkdir(basedir + "/train")
    if not os.path.exists(basedir + "/test"):
        os.mkdir(basedir + "/test")

    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)

df.to_csv(basedir + "/task-statics.csv")

path = basedir + '/test/'
all_test_data = {}
for client_id in range(num_client):
    for task in range(num_task):
        file = path + 'client-' + str(client_id) + '-task-' + str(task) + '.npz'
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
            if 'x' not in all_test_data.keys():
                all_test_data['x'] = data['x']
                all_test_data['y'] = data['y']
            else:
                all_test_data['x'] = np.concatenate((all_test_data['x'], data['x']))
                all_test_data['y'] = np.concatenate((all_test_data['y'], data['y']))

test_path = basedir + "/test/test-data"

with open(test_path + '.npz', 'wb') as f:
    np.savez_compressed(f, data=(all_test_data))

print("Dataset preparation completed!")
