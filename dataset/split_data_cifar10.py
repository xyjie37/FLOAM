import os.path
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from data_util import split_data, save_file

num_client = 20
num_task = 10  # You can set this to either 5 or 10
num_classes = 10  # CIFAR-10 has 10 classes
alpha = 0.1
np.random.seed(2266)

# Dataset storage path
datasetroot_dir = "/root/cifar10"
# Output dataset path
basedir = "./cifar10-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Define transforms to apply to the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                 std=[0.2470, 0.2435, 0.2616])])

# Download and load the CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root=datasetroot_dir, train=True, download=True, transform=transform)

# Download and load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root=datasetroot_dir, train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=len(test_dataset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    train_dataset.data, train_dataset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    test_dataset.data, test_dataset.targets = test_data

total_image = []
total_label = []

total_image.extend(train_dataset.data.cpu().detach().numpy())
total_image.extend(test_dataset.data.cpu().detach().numpy())
total_image = np.array(total_image)

total_label.extend(train_dataset.targets.cpu().detach().numpy())
total_label.extend(test_dataset.targets.cpu().detach().numpy())
total_label = np.array(total_label)

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
df = pd.DataFrame(columns=[str(i) for i in range(10)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_image[idxs]
    label_per_client[client] = total_label[idxs]
    row = [0 for i in range(10)]
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

col = [str(i) for i in range(10)]
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

    # Class-incremental mode
    classes_per_task = num_classes // num_task
    for task in range(num_task):
        start_class = 0
        end_class = 9
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
        row = [0 for i in range(12)]
        row[10] = client_id
        row[11] = task_id
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
