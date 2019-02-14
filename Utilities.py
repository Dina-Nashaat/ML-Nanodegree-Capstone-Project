import zipfile
import pandas as pd
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def visualize_bar(value_counts, title, xlabel, kind, ax):
    ax = value_counts.plot(kind=kind, figsize=(10,7), fontsize=13, ax=ax);
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13);
    for i in ax.patches:
        ax.text(i.get_width() + .1, i.get_y() + .31, str(round((i.get_width()), 2)), fontsize=13)

def load_dataset(dataset_type, dataset_path):
    zf = zipfile.ZipFile(dataset_path + dataset_type + '.csv.zip')
    return pd.read_csv(zf.open(dataset_type + '.csv'))

def undersample(undersample_target, keep_target, data, sample_ratio):
    n = int(len(data) * sample_ratio)
    undersampled_data = data[data['target'] == undersample_target].sample(n,random_state = 42)
    undersampled_data = undersampled_data.append(data[data['target'] == keep_target])
    return undersampled_data

def tokenize_and_pad(document, max_sequence_length, include_word_index=False):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(document)
    sequences = tokenizer.texts_to_sequences(document)
    padded = pad_sequences(sequences, maxlen=max_sequence_length)
    if include_word_index:
        word_index = tokenizer.word_index
        return word_index, padded
    return padded

def get_metrics(true_labels, predicted_labels, feature):
    print(feature)

    print('Accuracy:', metrics.accuracy_score(true_labels, predicted_labels))
    print('Precision:', metrics.precision_score(true_labels, predicted_labels))
    print('Recall:', metrics.recall_score(true_labels, predicted_labels))
    print('F1 Score:', metrics.f1_score(true_labels, predicted_labels))
    print('\n')