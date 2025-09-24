import numpy as np
import os
import torch
from torch.utils.data import DataLoader


def read_data(dataset, idx, task_num=0, is_train=True):
    if is_train:
        train_data_dir = os.path.join('' + dataset, 'train/')

        train_file = train_data_dir + 'client-' + str(idx) + '-task-' + str(task_num) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('' + dataset, 'test/')

        test_file = test_data_dir + 'client-' + str(idx) + '-task-' + str(task_num) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data
    
def read_all_test_data(dataset):
    test_data_dir = os.path.join('' + dataset, 'test/')

    test_file = test_data_dir + 'test-data.npz'
    with open(test_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()

    X_test = torch.Tensor(test_data['x']).type(torch.float32)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)
    
    # Add channel dimension for speech spectrograms
    if len(X_test.shape) == 3:  # If shape is (N, H, W), add channel dimension
        X_test = X_test.unsqueeze(1)  # Now shape is (N, 1, H, W)
    
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return DataLoader(test_data, 50, drop_last=False, shuffle=True)


def read_client_data(dataset, idx, task, is_train=True):

    if is_train:
        train_data = read_data(dataset, idx, task_num=task, is_train=True)
        # For text data (like yahooanswers), X should be integer token IDs, not float
        if dataset in ['yahooanswers', 'agnews', 'yelp', 'massiveproduct', 'dbpedia', 'newscategory']:
            X_train = torch.Tensor(train_data['x']).type(torch.long)
        else:
            X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        
        # Add channel dimension for speech spectrograms (batch_size, height, width) -> (batch_size, 1, height, width)
        if len(X_train.shape) == 3:  # If shape is (N, H, W), add channel dimension
            X_train = X_train.unsqueeze(1)  # Now shape is (N, 1, H, W)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, task_num=task, is_train=False)
        # For text data (like yahooanswers), X should be integer token IDs, not float
        if dataset in ['yahooanswers', 'agnews', 'yelp', 'massiveproduct', 'dbpedia', 'newscategory']:
            X_test = torch.Tensor(test_data['x']).type(torch.long)
        else:
            X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        
        # Add channel dimension for speech spectrograms
        if len(X_test.shape) == 3:  # If shape is (N, H, W), add channel dimension
            X_test = X_test.unsqueeze(1)  # Now shape is (N, 1, H, W)
        
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def load_train_data(dataset, id, task, batch_size=None):
        if batch_size == None:
            batch_size = 50
        train_data = read_client_data(dataset, id, task, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

def load_test_data(dataset, task, id, batch_size=None):
    if batch_size == None:
        batch_size = 16
    test_data = read_client_data(dataset, id, task=task, is_train=False)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)


def main(dataset_dir, num_clients, num_tasks, batch_size=50):
    print(f"Loading training data from: {dataset_dir}")
    
    for client_id in range(num_clients):
        for task_id in range(num_tasks):
            try:
                print(f"\nLoading Client {client_id}, Task {task_id}...")
                train_loader = load_train_data(dataset_dir, client_id, task_id, batch_size=batch_size)

                # 可选：打印一些样本信息
                first_batch = next(iter(train_loader))
                print(f"Batch size: {batch_size}")
                print(f"Number of batches: {len(train_loader)}")
                print(f"First batch X shape: {first_batch[0].shape}")
                print(f"First batch Y shape: {first_batch[1].shape}")

            except Exception as e:
                print(f"[ERROR] Failed to load client {client_id}, task {task_id}: {str(e)}")
                

if __name__ == "__main__":
    # 数据集路径（请替换为你的实际路径）
    dataset_dir = "/root/fedacd/dataset/cifar100-incremental-0.1-task-10"

    num_clients = 20
    num_tasks = 10

    # 每个 batch 的大小
    batch_size = 32

    # 启动主函数
    main(dataset_dir, num_clients, num_tasks, batch_size)
