import os
import pandas as pd
import numpy as np
import pickle
import string
from collections import Counter
from typing import List

from data_util import split_data, save_file

num_client = 5
num_task = 10
alpha = 0.3
np.random.seed(2266)

datasetroot_dir = "./20newsgroup"
train_file = os.path.join(datasetroot_dir, "train.csv")
test_file = os.path.join(datasetroot_dir, "test.csv")

basedir = "./20newsgroup-dir-{}-task-{}".format(alpha, num_task)
os.makedirs(basedir, exist_ok=True)

PREDEFINED_CLASS_NAMES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, encoding="latin1", encoding_errors="ignore")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.fillna("")


def _infer_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "labels", "class", "target", "category", "y"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


TEXT_COL_KEYWORDS = [
    "text",
    "content",
    "body",
    "message",
    "question",
    "title",
    "description",
    "summary",
]


def _select_text_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    non_label_cols = [col for col in df.columns if col != label_col]
    if not non_label_cols:
        raise ValueError("No text columns found in dataset")
    filtered = [col for col in non_label_cols if "label" not in col.lower()]
    if filtered:
        non_label_cols = filtered
    object_cols = [col for col in non_label_cols if pd.api.types.is_object_dtype(df[col])]
    preferred = [
        col
        for col in object_cols
        if any(keyword in col.lower() for keyword in TEXT_COL_KEYWORDS)
    ]
    if preferred:
        return preferred
    return object_cols if object_cols else non_label_cols


def _combine_text_fields(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
    def join_fields(row):
        parts = []
        for col in text_columns:
            value = row[col]
            if pd.isna(value):
                continue
            value = str(value).strip()
            if value and value.lower() != "nan":
                parts.append(value)
        return " ".join(parts)

    combined = df.apply(join_fields, axis=1)
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def load_20newsgroups_data(train_path, test_path, max_features=20000, max_len=400):
    print("Loading 20 Newsgroups dataset...")

    train_df = _load_csv(train_path)
    test_df = _load_csv(test_path)

    label_col = _infer_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in test data")

    label_text_col = None
    for candidate in ["label_text", "label_name", "category_text", "category_name"]:
        if candidate in train_df.columns and candidate != label_col:
            label_text_col = candidate
            break
    if label_text_col and label_text_col not in test_df.columns:
        label_text_col = None

    text_columns = _select_text_columns(train_df, label_col)
    train_df["text"] = _combine_text_fields(train_df, text_columns)
    test_df["text"] = _combine_text_fields(test_df, text_columns)

    train_df["text"] = train_df["text"].apply(preprocess_text)
    test_df["text"] = test_df["text"].apply(preprocess_text)

    combined_labels = pd.concat([train_df[label_col], test_df[label_col]], ignore_index=True)
    numeric_labels = pd.to_numeric(combined_labels, errors="coerce")
    if numeric_labels.notna().all():
        base_labels = numeric_labels.astype(int)
        unique_labels = sorted(base_labels.unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        if label_text_col:
            name_map = {}
            train_numeric = pd.to_numeric(train_df[label_col], errors="coerce")
            for raw_label, name in zip(train_numeric, train_df[label_text_col].astype(str)):
                if pd.notna(raw_label):
                    name_map[int(raw_label)] = name
            class_names = [name_map.get(label, str(label)) for label in unique_labels]
        else:
            class_names = (
                PREDEFINED_CLASS_NAMES
                if len(unique_labels) == len(PREDEFINED_CLASS_NAMES)
                else [str(label) for label in unique_labels]
            )
        train_labels = pd.to_numeric(train_df[label_col], errors="coerce").astype(int).map(label_to_idx).to_numpy(dtype=np.int64)
        test_labels = pd.to_numeric(test_df[label_col], errors="coerce").astype(int).map(label_to_idx).to_numpy(dtype=np.int64)
    else:
        combined_labels = combined_labels.astype(str)
        unique_labels = sorted(combined_labels.unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        if label_text_col:
            name_map = {}
            for raw_label, name in zip(train_df[label_col].astype(str), train_df[label_text_col].astype(str)):
                name_map[raw_label] = name
            class_names = [name_map.get(label, label) for label in unique_labels]
        else:
            class_names = (
                PREDEFINED_CLASS_NAMES
                if set(unique_labels) == set(PREDEFINED_CLASS_NAMES)
                else unique_labels
            )
        train_labels = train_df[label_col].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)
        test_labels = test_df[label_col].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    all_texts = train_df["text"].tolist() + test_df["text"].tolist()
    vocab, word_to_idx = _build_vocabulary(all_texts, max_features)

    X_train = _encode_texts(train_df["text"].tolist(), word_to_idx, max_len)
    X_test = _encode_texts(test_df["text"].tolist(), word_to_idx, max_len)

    vocab_path = os.path.join(basedir, "vocabulary.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({"vocab": vocab, "word_to_idx": word_to_idx}, f)

    return X_train, train_labels, X_test, test_labels, len(vocab), class_names


def _build_vocabulary(texts, max_features):
    print("Building vocabulary...")
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    vocab = ["<PAD>", "<UNK>"] + [word for word, _ in word_counts.most_common(max_features - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_idx


def _encode_texts(texts, word_to_idx, max_len):
    sequences = np.zeros((len(texts), max_len), dtype=np.int64)

    pad_idx = word_to_idx["<PAD>"]
    unk_idx = word_to_idx["<UNK>"]

    for idx, text in enumerate(texts):
        words = text.split()[:max_len]
        token_ids = [word_to_idx.get(word, unk_idx) for word in words]
        length = len(token_ids)
        sequences[idx, :length] = token_ids
        if length < max_len:
            sequences[idx, length:] = pad_idx

    return sequences


X_train, y_train, X_test, y_test, vocab_size, class_names = load_20newsgroups_data(train_file, test_file)
num_classes = len(class_names)

print(f"20 Newsgroups dataset with {num_classes} classes")
print(f"Classes: {class_names}")

print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
print(f"Vocabulary size: {vocab_size}")

# Combine training and test data
total_data = np.concatenate([X_train, X_test], axis=0)
total_label = np.concatenate([y_train, y_test], axis=0)

print(f"Total data shape: {total_data.shape}")
print(f"Total labels shape: {total_label.shape}")
print(f"Classes distribution: {np.bincount(total_label)}")

image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]

dataidx_map = {}
idxs = np.array(range(len(total_label)))
idx_for_each_class = [idxs[total_label == i] for i in range(num_classes)]

for i in range(num_classes):
    num_samples = len(idx_for_each_class[i])
    num_per_client = num_samples / num_client
    per_client_sample_number = [int(num_per_client) for _ in range(num_client)]
    remainder = num_samples - sum(per_client_sample_number)
    for j in range(remainder):
        per_client_sample_number[j] += 1

    idx = 0
    for client, num_sample in enumerate(per_client_sample_number):
        if client not in dataidx_map:
            dataidx_map[client] = idx_for_each_class[i][idx : idx + num_sample]
        else:
            dataidx_map[client] = np.append(
                dataidx_map[client], idx_for_each_class[i][idx : idx + num_sample], axis=0
            )
        idx += num_sample

df = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_data[idxs]
    label_per_client[client] = total_label[idxs]
    row = [0 for _ in range(num_classes)]
    for i in np.unique(label_per_client[client]):
        statistic[client].append((int(i), int(sum(label_per_client[client] == i))))
        row[i] = int(sum(label_per_client[client] == i))
    df.loc[len(df)] = row

df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    print(
        f"Client {client}\t Size of data: {len(image_per_client[client])}\t Labels: ",
        np.unique(label_per_client[client]),
    )
    print("\t\t Samples of labels: ", [i for i in statistic[client]])
    print("=" * 50)

least_samples = len(image_per_client[0]) // 10
if num_task == 10:
    least_samples = len(image_per_client[0]) // 20
print("least samples:", least_samples)

col = [str(i) for i in range(num_classes)]
col.append("client")
col.append("task")
df = pd.DataFrame(columns=col)

for client_id in range(num_client):
    client_data = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    for task in range(num_task):
        task_classes = list(range(0, num_classes))
        idx_batch = [[] for _ in range(num_task)]
        for k in task_classes:
            idx_k = np.where(client_dataset_label == k)[0]
            if len(idx_k) > 0:
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_task))
                proportions = np.clip(proportions, a_min=0.05, a_max=None)
                proportions /= proportions.sum()
                splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, splits))
                ]
        for j in range(num_task):
            client_idx_map[j] = idx_batch[j]

    for task_id in range(num_task):
        row = [0 for _ in range(num_classes + 2)]
        row[num_classes] = client_id
        row[num_classes + 1] = task_id
        idxs = client_idx_map[task_id]
        Y[task_id] = client_dataset_label[idxs].astype(np.int64)
        X[task_id] = client_data[idxs].astype(np.int64)

        info = []
        for i in np.unique(Y[task_id]):
            info.append((int(i), int(sum(Y[task_id] == i))))
            row[i] = int(sum(Y[task_id] == i))
        df.loc[len(df)] = row

        print(
            f"Client {client_id}  Task {task_id}\t Size of data: {len(X[task_id])}\t Labels: ",
            np.unique(Y[task_id]),
        )
        print("\t\t Samples of labels: ", [i for i in info])
        print("-" * 50)

    print("=" * 50 + "\n\n")

    train_data, test_data = split_data(X, Y)

    os.makedirs(basedir + "/train", exist_ok=True)
    os.makedirs(basedir + "/test", exist_ok=True)

    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)

df.to_csv(basedir + "/task-statics.csv")

path = basedir + "/test/"
all_test_data = {}
for client_id in range(num_client):
    for task in range(num_task):
        file = path + "client-" + str(client_id) + "-task-" + str(task) + ".npz"
        with open(file, "rb") as f:
            data = np.load(f, allow_pickle=True)["data"].tolist()
            if "x" not in all_test_data:
                all_test_data["x"] = data["x"]
                all_test_data["y"] = data["y"]
            else:
                all_test_data["x"] = np.concatenate((all_test_data["x"], data["x"]))
                all_test_data["y"] = np.concatenate((all_test_data["y"], data["y"]))

test_path = basedir + "/test/test-data"
with open(test_path + ".npz", "wb") as f:
    np.savez_compressed(f, data=(all_test_data))

print("Dataset split completed!")
print(f"Files saved to: {basedir}")
print(f"Vocabulary size: {vocab_size}")
print(f"Number of clients: {num_client}")
print(f"Number of tasks: {num_task}")
print(f"Number of classes: {num_classes}")
