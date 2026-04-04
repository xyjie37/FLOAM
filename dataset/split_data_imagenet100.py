"""
ImageNet100 data allocation script.
Follows the same format as other dataset split scripts.
Expected structure: images/train/class_name/*.jpg and images/val/class_name/*.jpg
"""
import os.path
import gc
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_util import split_data, save_file

num_client = 20
num_task = 10
num_classes = 100
alpha = 0.1
np.random.seed(2266)

# Dataset root path (configurable)
datasetroot_dir = "/home/jxy/datasets/imagenet100/images"
# Output directory
basedir = "./imagenet100-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# ImageNet normalization, resize to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Number of workers for parallel loading (adjust based on CPU cores)
NUM_WORKERS = 8


def load_image(path):
    """Load and transform a single image."""
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
        return transform(img).numpy()
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}") from e


def load_images_batch(paths, max_workers):
    """Load images in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_image, p): i for i, p in enumerate(paths)}
        results = [None] * len(paths)
        for future in tqdm(as_completed(futures), total=len(paths), desc="Loading", unit="img"):
            idx = futures[future]
            results[idx] = future.result()
    return np.array(results)


class ImageNet100Dataset:
    """Loader for ImageNet100 with standard layout: train/class_name/*.jpg, val/class_name/*.jpg"""

    def __init__(self, root_dir, mode='train'):
        self.images = []
        self.labels = []

        if mode == 'train':
            path = os.path.join(root_dir, 'train')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Train path not found: {path}")
            classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            for cls in classes:
                cls_path = os.path.join(path, cls)
                for img in os.listdir(cls_path):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(cls_path, img)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[cls])
        else:
            path = os.path.join(root_dir, 'val')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Val path not found: {path}")
            train_path = os.path.join(root_dir, 'train')
            classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            for cls in classes:
                cls_path = os.path.join(path, cls)
                if os.path.isdir(cls_path):
                    for img in os.listdir(cls_path):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(cls_path, img)
                            self.images.append(img_path)
                            self.labels.append(self.class_to_idx[cls])


# Load dataset paths and labels only (no image data - saves memory)
print("Loading dataset paths...")
train_dataset = ImageNet100Dataset(datasetroot_dir, 'train')
test_dataset = ImageNet100Dataset(datasetroot_dir, 'val')

total_image_paths = train_dataset.images + test_dataset.images
total_label = np.array(train_dataset.labels + test_dataset.labels)
del train_dataset, test_dataset
gc.collect()

# Build client index mapping (no images loaded yet)
dataidx_map = {}
idxs = np.array(range(len(total_label)))
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_label == i])

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

# Client statistics (from labels only, no images)
label_per_client = [total_label[dataidx_map[c]] for c in range(num_client)]
statistic = [[] for _ in range(num_client)]
df = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    row = [0 for i in range(num_classes)]
    for i in np.unique(label_per_client[client]):
        cnt = int(sum(label_per_client[client] == i))
        statistic[client].append((int(i), cnt))
        row[i] = cnt
    df.loc[len(df)] = row
df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    print(f"Client {client}\t Size of data: {len(label_per_client[client])}\t Labels: ", np.unique(label_per_client[client]))
    print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    print("=" * 50)

K = num_classes
least_samples = len(label_per_client[0]) // 10
if num_task == 10:
    least_samples = len(label_per_client[0]) // 20
print("least samples:", least_samples)
N = least_samples

col = [str(i) for i in range(num_classes)]
col.append('client')
col.append('task')
df = pd.DataFrame(columns=col)

# Dirichlet allocation across tasks - load images per client to save memory
if not os.path.exists(basedir + "/train"):
    os.mkdir(basedir + "/train")
if not os.path.exists(basedir + "/test"):
    os.mkdir(basedir + "/test")

for client_id in range(num_client):
    # Load only this client's images (memory-efficient: 1/num_client of full dataset)
    client_indices = dataidx_map[client_id]
    client_paths = [total_image_paths[i] for i in client_indices]
    print(f"Client {client_id}: Loading {len(client_paths)} images...")
    client_images = load_images_batch(client_paths, NUM_WORKERS)
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    task_classes = list(range(num_classes))
    idx_batch = [[] for _ in range(num_task)]
    for k in task_classes:
        idx_k = np.where(client_dataset_label == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_task))
        proportions = np.clip(proportions, a_min=0.05, a_max=None)
        proportions /= proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

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

    # Save data
    train_data, test_data = split_data(X, Y)
    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)

    # Free memory before next client
    del client_images, X, Y, train_data, test_data, client_paths
    gc.collect()

df.to_csv(basedir + "/task-statics.csv")

# Aggregate test data (use list append + single concat to avoid memory bloat)
print("Aggregating test data...")
path = basedir + '/test/'
all_test_x, all_test_y = [], []
for client_id in tqdm(range(num_client), desc="Aggregating", unit="client"):
    for task in range(num_task):
        file = path + 'client-' + str(client_id) + '-task-' + str(task) + '.npz'
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
            all_test_x.append(data['x'])
            all_test_y.append(data['y'])
all_test_data = {'x': np.concatenate(all_test_x), 'y': np.concatenate(all_test_y)}
del all_test_x, all_test_y
gc.collect()

test_path = basedir + "/test/test-data"
with open(test_path + '.npz', 'wb') as f:
    np.savez_compressed(f, data=(all_test_data))
del all_test_data
gc.collect()

print("Dataset preparation completed!")
