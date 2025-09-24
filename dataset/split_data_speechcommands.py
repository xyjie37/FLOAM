import os.path
import pandas as pd
import numpy as np
import torch
import torchaudio
# Use soundfile library to load WAV files without sox dependency
import soundfile as sf  # type: ignore
def soundfile_load(path):
    data, samplerate = sf.read(path, dtype='float32')
    if data.ndim == 1:
        data = np.expand_dims(data, 1)
    # Convert to (channels, time)
    data = torch.from_numpy(data.T)
    return data, samplerate
torchaudio.load = soundfile_load
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS

from data_util import split_data, save_file

num_client = 20
num_task = 10  # You can set this to either 5 or 10
num_classes = 30  # Speech Commands v1 has 30 classes
alpha = 0.1
np.random.seed(2266)

# 数据集存储地址
datasetroot_dir = "./speechcommands"
# 生成数据集存储地址
basedir = "./speechcommands-dir-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Speech Commands v1 classes (30 classes)
SPEECH_COMMANDS_V1_CLASSES = [
    'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
    'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero'
]

class SpeechCommandsSubset(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = None):
        super().__init__(root=root, download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        
        # Filter for v1 classes only
        self._walker = [w for w in self._walker if self._get_label(w) in SPEECH_COMMANDS_V1_CLASSES]

    def _get_label(self, filepath):
        return os.path.dirname(filepath).split(os.path.sep)[-1]

# Download and prepare datasets
print("Loading Speech Commands dataset...")
if not os.path.exists(datasetroot_dir):
    os.makedirs(datasetroot_dir)
train_dataset = SpeechCommandsSubset(subset="training", root=datasetroot_dir)
val_dataset = SpeechCommandsSubset(subset="validation", root=datasetroot_dir)
test_dataset = SpeechCommandsSubset(subset="testing", root=datasetroot_dir)

# Combine training and validation for our federated setup
all_datasets = [train_dataset, val_dataset, test_dataset]

# Audio preprocessing parameters
sample_rate = 16000
n_mels = 32  # 改为32个mel频率bins
n_fft = 2048
hop_length = 512

# Define audio transforms
transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    power=2.0,
)

# Convert to log scale
to_db = T.AmplitudeToDB()

def preprocess_audio(waveform, sample_rate_orig):
    # Resample if necessary
    if sample_rate_orig != sample_rate:
        resampler = T.Resample(sample_rate_orig, sample_rate)
        waveform = resampler(waveform)
    
    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Pad or truncate to fixed length (1 second)
    target_length = sample_rate
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Convert to mel spectrogram
    mel_spec = transform(waveform)
    mel_spec = to_db(mel_spec)
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    return mel_spec.squeeze(0).numpy()

print("Processing audio data...")
total_data = []
total_labels = []

# Create label mapping
label_to_idx = {label: idx for idx, label in enumerate(SPEECH_COMMANDS_V1_CLASSES)}

# Process all datasets
for dataset in all_datasets:
    for i, (waveform, sample_rate_orig, label, speaker_id, utterance_number) in enumerate(dataset):
        if label in SPEECH_COMMANDS_V1_CLASSES:
            processed_audio = preprocess_audio(waveform, sample_rate_orig)
            total_data.append(processed_audio)
            total_labels.append(label_to_idx[label])
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples...")

total_data = np.array(total_data)
total_labels = np.array(total_labels)

print(f"Total samples: {len(total_data)}")
print(f"Data shape: {total_data.shape}")
print(f"Unique labels: {np.unique(total_labels)}")

image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]

# 记录索引的字典 key = client编号  value = []索引list
dataidx_map = {}
# 每一个数据的索引
idxs = np.array(range(len(total_labels)))
# 每一个类数据的索引
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_labels == i])

# 对每类数据操作
for i in range(num_classes):
    num_images = len(idx_for_each_class[i])
    if num_images == 0:
        continue
    num_per_client = num_images / num_client
    per_client_image_number = [int(num_per_client) for _ in range(num_client)]
    
    # Handle remainder
    remainder = num_images - sum(per_client_image_number)
    for j in range(remainder):
        per_client_image_number[j] += 1
    
    idx = 0
    for client, num_sample in enumerate(per_client_image_number):
        if num_sample == 0:
            continue
        if client not in dataidx_map.keys():
            dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
        else:
            dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
        idx += num_sample

# 遍历每个客户端,得到每个客户端的索引
df = pd.DataFrame(columns=[str(i) for i in range(num_classes)])
for client in range(num_client):
    if client in dataidx_map:
        idxs = dataidx_map[client]
        image_per_client[client] = total_data[idxs]
        label_per_client[client] = total_labels[idxs]
        row = [0 for i in range(num_classes)]
        for i in np.unique(label_per_client[client]):
            statistic[client].append((int(i), int(sum(label_per_client[client] == i))))
            row[i] = int(sum(label_per_client[client] == i))
        df.loc[len(df)] = row
    else:
        # Client has no data
        image_per_client[client] = np.array([])
        label_per_client[client] = np.array([])
        df.loc[len(df)] = [0] * num_classes

df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    if len(image_per_client[client]) > 0:
        print(f"Client {client}\t Size of data: {len(image_per_client[client])}\t Labels: ", np.unique(label_per_client[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    else:
        print(f"Client {client}\t Size of data: 0\t Labels: []")
    print("=" * 50)

# Calculate minimum samples for task division
non_empty_clients = [i for i in range(num_client) if len(image_per_client[i]) > 0]
if non_empty_clients:
    least_samples = min([len(image_per_client[i]) for i in non_empty_clients]) // 10
else:
    least_samples = 0

if num_task == 10:
    least_samples = least_samples // 2 if least_samples > 0 else 0
print("least samples:", least_samples)

col = [str(i) for i in range(num_classes)]
col.append('client')
col.append('task')
df = pd.DataFrame(columns=col)

# 将每类数据按照迪利克雷分布分配给每个任务
for client_id in range(num_client):
    if len(image_per_client[client_id]) == 0:
        # Skip clients with no data
        for task_id in range(num_task):
            row = [0 for i in range(num_classes + 2)]
            row[num_classes] = client_id
            row[num_classes + 1] = task_id
            df.loc[len(df)] = row
        continue
        
    client_images = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    # 类增量模式
    task_classes = list(range(0, num_classes))  # All classes for all tasks

    idx_batch = [[] for _ in range(num_task)]
    for k in task_classes:
        idx_k = np.where(client_dataset_label == k)[0]
        if len(idx_k) == 0:
            continue
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_task))
        proportions = np.clip(proportions, a_min=0.05, a_max=None)  # 确保最小比例
        proportions /= proportions.sum()  # 归一化
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_splits = np.split(idx_k, proportions)
        for j in range(min(num_task, len(idx_splits))):
            idx_batch[j].extend(idx_splits[j].tolist())

    for j in range(num_task):
        client_idx_map[j] = idx_batch[j]

    for task_id in range(num_task):
        row = [0 for i in range(num_classes + 2)]
        row[num_classes] = client_id
        row[num_classes + 1] = task_id
        idxs = client_idx_map[task_id]
        
        if len(idxs) > 0:
            Y[task_id] = client_dataset_label[idxs]
            X[task_id] = client_images[idxs]

            info = []
            for i in np.unique(Y[task_id]):
                info.append((int(i), int(sum(Y[task_id] == i))))
                row[i] = int(sum(Y[task_id] == i))

            print(f"Client {client_id}  Task {task_id}\t Size of data: {len(X[task_id])}\t Labels: ", np.unique(Y[task_id]))
            print(f"\t\t Samples of labels: ", [i for i in info])
        else:
            Y[task_id] = np.array([])
            X[task_id] = np.array([]).reshape(0, *client_images.shape[1:])
            print(f"Client {client_id}  Task {task_id}\t Size of data: 0\t Labels: []")
        
        df.loc[len(df)] = row
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

# Aggregate all test data
path = basedir + '/test/'
all_test_data = {}
for client_id in range(num_client):
    for task in range(num_task):
        file = path + 'client-' + str(client_id) + '-task-' + str(task) + '.npz'
        try:
            with open(file, 'rb') as f:
                data = np.load(f, allow_pickle=True)['data'].tolist()
                if len(data['x']) > 0:  # Only add non-empty data
                    if 'x' not in all_test_data.keys():
                        all_test_data['x'] = data['x']
                        all_test_data['y'] = data['y']
                    else:
                        all_test_data['x'] = np.concatenate((all_test_data['x'], data['x']))
                        all_test_data['y'] = np.concatenate((all_test_data['y'], data['y']))
        except FileNotFoundError:
            continue

test_path = basedir + "/test/test-data"

if 'x' in all_test_data and len(all_test_data['x']) > 0:
    with open(test_path + '.npz', 'wb') as f:
        np.savez_compressed(f, data=(all_test_data))
    print(f"Saved aggregated test data with {len(all_test_data['x'])} samples")
else:
    print("No test data to aggregate")

print(f"Dataset generation completed. Data saved in {basedir}")
print(f"Speech Commands classes used: {SPEECH_COMMANDS_V1_CLASSES}")
