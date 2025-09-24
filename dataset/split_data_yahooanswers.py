import os
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string
from collections import Counter

from data_util import split_data, save_file

num_client = 20
num_task = 5
num_classes = 10  # Yahoo! Answers has 10 classes
alpha = 0.1
np.random.seed(2266)

# Yahoo! Answers class names
class_names = [
    'Society & Culture',
    'Science & Mathematics', 
    'Health',
    'Education & Reference',
    'Computers & Internet',
    'Sports',
    'Business & Finance',
    'Entertainment & Music',
    'Family & Relationships',
    'Politics & Government'
]

print(f"Yahoo! Answers dataset with {num_classes} classes")
print(f"Classes: {class_names}")

datasetroot_dir = "./yahoo_answers"
train_file = os.path.join(datasetroot_dir, "train.csv") 
test_file = os.path.join(datasetroot_dir, "test.csv")

# 生成数据集存储地址
basedir = "./yahooanswers-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

def preprocess_text(text):
    """
    Simple text preprocessing
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_yahoo_answers_data(train_file, test_file, max_features=10000, max_len=200):
    """
    Load and preprocess Yahoo! Answers dataset
    Expected format: CSV with columns [class, question_title, question_content, best_answer]
    """
    print("Loading Yahoo! Answers dataset...")
    
    # Load training data
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, header=None, 
                              names=['class', 'question_title', 'question_content', 'best_answer'])
    else:
        print(f"Warning: {train_file} not found. Using dummy data for demonstration.")
        exit
    
    # Load test data
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file, header=None,
                             names=['class', 'question_title', 'question_content', 'best_answer'])
    else:
        print(f"Warning: {test_file} not found. Using dummy data for demonstration.")
        # Create dummy data for demonstration - increased size to simulate real dataset
        n_samples = 60000  # 模拟真实Yahoo! Answers测试集大小
        test_df = pd.DataFrame({
            'class': np.random.randint(1, num_classes + 1, n_samples),
            'question_title': [f"Test question {i}" for i in range(n_samples)],
            'question_content': [f"Test content {i}" for i in range(n_samples)],
            'best_answer': [f"Test answer {i}" for i in range(n_samples)]
        })
    
    # Combine text fields
    train_df['text'] = train_df['question_title'].fillna('') + ' ' + \
                      train_df['question_content'].fillna('') + ' ' + \
                      train_df['best_answer'].fillna('')
    
    test_df['text'] = test_df['question_title'].fillna('') + ' ' + \
                     test_df['question_content'].fillna('') + ' ' + \
                     test_df['best_answer'].fillna('')
    
    # Preprocess text
    train_df['text'] = train_df['text'].apply(preprocess_text)
    test_df['text'] = test_df['text'].apply(preprocess_text)
    
    # Convert class labels to 0-based indexing
    train_df['class'] = train_df['class'] - 1
    test_df['class'] = test_df['class'] - 1
    
    # Combine all text for building vocabulary
    all_texts = list(train_df['text']) + list(test_df['text'])
    
    # Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter()
    for text in all_texts:
        words = text.split()
        word_counts.update(words)
    
    # Get most common words
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(max_features - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    def text_to_sequence(text, word_to_idx, max_len):
        """Convert text to sequence of word indices"""
        words = text.split()[:max_len]
        sequence = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
        # Pad sequence
        sequence += [word_to_idx['<PAD>']] * (max_len - len(sequence))
        return sequence[:max_len]
    
    # Convert text to sequences
    print("Converting text to sequences...")
    train_sequences = [text_to_sequence(text, word_to_idx, max_len) for text in train_df['text']]
    test_sequences = [text_to_sequence(text, word_to_idx, max_len) for text in test_df['text']]
    
    # Convert to numpy arrays with explicit integer dtype for text data
    X_train = np.array(train_sequences, dtype=np.int64)
    y_train = np.array(train_df['class'], dtype=np.int64)
    X_test = np.array(test_sequences, dtype=np.int64)
    y_test = np.array(test_df['class'], dtype=np.int64)
    
    print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary for later use
    vocab_path = os.path.join(basedir, "vocabulary.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump({'vocab': vocab, 'word_to_idx': word_to_idx}, f)
    
    return X_train, y_train, X_test, y_test, len(vocab)

# Load and preprocess data
X_train, y_train, X_test, y_test, vocab_size = load_yahoo_answers_data(train_file, test_file)

# Combine training and test data
total_data = np.concatenate([X_train, X_test], axis=0)
total_label = np.concatenate([y_train, y_test], axis=0)

print(f"Total data shape: {total_data.shape}")
print(f"Total labels shape: {total_label.shape}")
print(f"Classes distribution: {np.bincount(total_label)}")

# Data distribution statistics
image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]

# 记录索引的字典 key = client编号  value = []索引list
dataidx_map = {}
# 每一个数据的索引
idxs = np.array(range(len(total_label)))
# 每一个类数据的索引
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_label == i])

# 对每类数据操作 - 均匀分布给各个客户端
for i in range(num_classes):
    num_samples = len(idx_for_each_class[i])
    num_per_client = num_samples / num_client
    per_client_sample_number = [int(num_per_client) for _ in range(num_client)]
    
    # Handle remainder samples
    remainder = num_samples - sum(per_client_sample_number)
    for j in range(remainder):
        per_client_sample_number[j] += 1
    
    idx = 0
    for client, num_sample in enumerate(per_client_sample_number):
        if client not in dataidx_map.keys():
            dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
        else:
            dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
        idx += num_sample

# 遍历每个客户端,得到每个客户端的数据分布统计
df = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_data[idxs]
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

# 将每类数据按照迪利克雷分布分配给每个任务
for client_id in range(num_client):
    client_data = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    # 类增量模式 - 所有任务都包含所有类别
    for task in range(num_task):
        start_class = 0
        end_class = num_classes - 1
        task_classes = list(range(start_class, end_class + 1))

        idx_batch = [[] for _ in range(num_task)]
        for k in task_classes:
            idx_k = np.where(client_dataset_label == k)[0]
            if len(idx_k) > 0:
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_task))
                proportions = np.clip(proportions, a_min=0.05, a_max=None)  # 确保最小比例
                proportions /= proportions.sum()  # 归一化
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(num_task):
            client_idx_map[j] = idx_batch[j]

    for task_id in range(num_task):
        row = [0 for i in range(num_classes + 2)]
        row[num_classes] = client_id
        row[num_classes + 1] = task_id
        idxs = client_idx_map[task_id]
        Y[task_id] = client_dataset_label[idxs].astype(np.int64)
        X[task_id] = client_data[idxs].astype(np.int64)  # 确保文本数据保持为整数类型

        info = []
        for i in np.unique(Y[task_id]):
            info.append((int(i), int(sum(Y[task_id] == i))))
            row[i] = int(sum(Y[task_id] == i))
        df.loc[len(df)] = row

        print(f"Client {client_id}  Task {task_id}\t Size of data: {len(X[task_id])}\t Labels: ", np.unique(Y[task_id]))
        print(f"\t\t Samples of labels: ", [i for i in info])
        print("-" * 50)

    print("=" * 50 + "\n\n")

    # 保存数据
    train_data, test_data = split_data(X, Y)

    if not os.path.exists(basedir + "/train"):
        os.mkdir(basedir + "/train")
    if not os.path.exists(basedir + "/test"):
        os.mkdir(basedir + "/test")

    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)

df.to_csv(basedir + "/task-statics.csv")

# 合并所有测试数据
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

print(f"Dataset split completed!")
print(f"Files saved to: {basedir}")
print(f"Vocabulary size: {vocab_size}")
print(f"Number of clients: {num_client}")
print(f"Number of tasks: {num_task}")
print(f"Number of classes: {num_classes}")
