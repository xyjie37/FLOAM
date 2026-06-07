import os
import os.path
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from data_util import split_data, save_file

# 1. Parameters
num_client = 20
num_task = 10
num_classes = 200
alpha = 0.5  # Dirichlet parameter controlling client heterogeneity
np.random.seed(2266)

# Dataset storage path
datasetroot_dir = "/root/tiny-imagenet-200"
# Output dataset path
basedir = "./tinyimagenet-incremental-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.makedirs(basedir)

# 2. Tiny-ImageNet dataset loading
class TinyImageNetDataset:
    """Custom TinyImageNet dataset loader"""
    def __init__(self, root_dir, mode='train', class_to_idx=None):
        self.root_dir = root_dir
        self.mode = mode
        self.images = []
        self.labels = []
        
        if class_to_idx is None:
            train_path = os.path.join(self.root_dir, 'train')
            classes = sorted(os.listdir(train_path))
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        if self.mode == 'train':
            path = os.path.join(self.root_dir, 'train')
            for cls in self.class_to_idx.keys():
                cls_path = os.path.join(path, cls, 'images')
                for img_name in os.listdir(cls_path):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])
        else: # mode == 'val'
            path = os.path.join(self.root_dir, 'val')
            val_annotations_path = os.path.join(path, 'val_annotations.txt')
            img_to_cls = {}
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_to_cls[parts[0]] = parts[1]
            
            val_images_path = os.path.join(path, 'images')
            for img_name in os.listdir(val_images_path):
                if img_name in img_to_cls:
                    cls = img_to_cls[img_name]
                    self.images.append(os.path.join(val_images_path, img_name))
                    self.labels.append(self.class_to_idx[cls])

print("Loading dataset paths...")
train_dataset = TinyImageNetDataset(datasetroot_dir, 'train')
test_dataset = TinyImageNetDataset(datasetroot_dir, 'val', class_to_idx=train_dataset.class_to_idx)
total_image_paths = train_dataset.images + test_dataset.images
total_label = np.array(train_dataset.labels + test_dataset.labels)

print("Loading all images (this may take a while and consume memory)...")
total_image = np.array([np.array(Image.open(p).convert('RGB')) for p in total_image_paths])
print(f"Total images loaded: {total_image.shape}")
print(f"Total labels loaded: {total_label.shape}")

# 3. Assign dataset to clients via Dirichlet distribution (Phase 1)
image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]
dataidx_map = {}

idxs = np.array(range(len(total_label)))
idx_for_each_class = [idxs[total_label == i] for i in range(num_classes)]

# Generate per-client sample distribution for each class
dirichlet_dist = np.random.dirichlet([alpha] * num_client, num_classes)

for i in range(num_classes):
    class_indices = idx_for_each_class[i]
    num_images = len(class_indices)
    
    # Compute per-client sample counts from Dirichlet proportions
    client_distribution = np.round(dirichlet_dist[i] * num_images).astype(int)
    # Fix rounding mismatch
    diff = num_images - client_distribution.sum()
    client_distribution[-1] += diff
    
    current_pos = 0
    for client in range(num_client):
        num_sample = client_distribution[client]
        assigned_indices = class_indices[current_pos : current_pos + num_sample]
        
        if client not in dataidx_map:
            dataidx_map[client] = assigned_indices
        else:
            dataidx_map[client] = np.append(dataidx_map[client], assigned_indices, axis=0)
        current_pos += num_sample

# Gather per-client data and record statistics
df_client = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    client_idxs = dataidx_map[client].astype(int)
    np.random.shuffle(client_idxs)
    
    image_per_client[client] = total_image[client_idxs]
    label_per_client[client] = total_label[client_idxs]
    
    row = [0] * num_classes
    unique_labels, counts = np.unique(label_per_client[client], return_counts=True)
    for label, count in zip(unique_labels, counts):
        statistic[client].append((int(label), int(count)))
        row[label] = int(count)
    df_client.loc[len(df_client)] = row

df_client.to_csv(os.path.join(basedir, "client-statics.csv"))

print("\n--- Client Data Distribution Summary (Non-IID) ---")
for client in range(num_client):
    print(f"Client {client}\t Size of data: {len(image_per_client[client])}\t Num of Classes: {len(np.unique(label_per_client[client]))}")
    print("=" * 60)

# 4. Split each client's data into class-incremental tasks (Phase 2)
df_task = pd.DataFrame(columns=[str(i) for i in range(num_classes)] + ['client', 'task'])
classes_per_task = num_classes // num_task

for client_id in range(num_client):
    print(f"\nProcessing Client {client_id} for Class-Incremental Tasks...")
    client_images = image_per_client[client_id]
    client_labels = label_per_client[client_id]
    
    X_tasks = [[] for _ in range(num_task)]
    Y_tasks = [[] for _ in range(num_task)]
    client_idx_map = {}

    # Split this client's data into tasks
    for task_id in range(num_task):
        # Define classes for current task
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))

        # Find samples in client data belonging to these classes
        task_idx = []
        for k in task_classes:
            idx_k = np.where(client_labels == k)[0]
            if len(idx_k) > 0:
                task_idx.extend(idx_k)
        
        client_idx_map[task_id] = np.array(task_idx)

    # Organize each task's data and record statistics
    for task_id in range(num_task):
        row = [0] * (num_classes + 2)
        row[num_classes] = client_id
        row[num_classes + 1] = task_id
        
        task_idxs = client_idx_map[task_id].astype(int)
        
        if len(task_idxs) > 0:
            Y_tasks[task_id] = client_labels[task_idxs]
            X_tasks[task_id] = client_images[task_idxs]
        else:
            Y_tasks[task_id] = np.array([])
            X_tasks[task_id] = np.array([])

        info = []
        unique_labels, counts = np.unique(Y_tasks[task_id], return_counts=True)
        for label, count in zip(unique_labels, counts):
            info.append((int(label), int(count)))
            row[int(label)] = int(count)
        df_task.loc[len(df_task)] = row

        print(f"Client {client_id} Task {task_id}\t Size: {len(X_tasks[task_id])}\t Labels: {np.unique(Y_tasks[task_id])}")
        print("-" * 60)

    # 5. Save this client's task data
    # Apply transform to image data
    X_tasks_transformed = []
    norm_transform = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    for task_images in X_tasks:
        if len(task_images) > 0:
            task_tensor = torch.from_numpy(task_images).permute(0, 3, 1, 2).float() / 255.0
            transformed_images = norm_transform(task_tensor).numpy()
            X_tasks_transformed.append(transformed_images)
        else:
            X_tasks_transformed.append(task_images)

    train_data, test_data = split_data(X_tasks_transformed, Y_tasks)

    train_dir = os.path.join(basedir, "train")
    test_dir = os.path.join(basedir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path_prefix = os.path.join(train_dir, f"client-{client_id}-task-")
    test_path_prefix = os.path.join(test_dir, f"client-{client_id}-task-")
    save_file(train_path_prefix, test_path_prefix, train_data, test_data)

df_task.to_csv(os.path.join(basedir, "task-statics.csv"))

# 6. Merge all clients' test data
print("\nAggregating all test data...")
path = os.path.join(basedir, 'test/')
all_test_data = {'x': [], 'y': []}
for client_id in range(num_client):
    for task_id in range(num_task):
        file_path = os.path.join(path, f'client-{client_id}-task-{task_id}.npz')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)['data'].item()
                if 'x' in data and data['x'].shape[0] > 0:
                    all_test_data['x'].append(data['x'])
                    all_test_data['y'].append(data['y'])

if all_test_data['x']:
    all_test_data['x'] = np.concatenate(all_test_data['x'], axis=0)
    all_test_data['y'] = np.concatenate(all_test_data['y'], axis=0)
    test_path = os.path.join(basedir, "test", "all_test_data.npz")
    with open(test_path, 'wb') as f:
        np.savez_compressed(f, data=all_test_data)
    print(f"Total aggregated test samples: {len(all_test_data['y'])}")
else:
    print("No test data was generated to aggregate.")

print(f"\nData generation complete. Files saved in: {basedir}")