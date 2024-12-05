import logging
import random

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from poutyne.framework import Model

import poutyne as pt


class WordClassifier(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.rnn = nn.RNN(embedding_size, hidden_size,
                            4, dropout = 0.5,
                            batch_first = True, bidirectional=False)
        self.mapping_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, inputs):
        # TODO:
        out1 = self.embedding(inputs)
        out2, _ = self.rnn(out1)
        output = self.mapping_layer(out2)
        
        return torch.mean(output, dim=1)


class WordClassifierHandlingPadding(WordClassifier):
    def __init__(self, vocab, embedding_size, hidden_size, num_classes):
        super().__init__(vocab, embedding_size, hidden_size, num_classes)

    def forward(self, inputs):
        # TODO: Gestion des exemples a longueur variable
        # Servez-vous de la fonction pack_padded_sequence

        len_list = []#[inputs.shape[1] for i in range(inputs.shape[0])]
        # # max_len = inputs.shape[1]
        for row in range(inputs.shape[0]):
            x = inputs[row, :]
            x_len = torch.where(x == 0)
            if len(x_len[0]) == 0:
                x_len = inputs.shape[1]
            else:
                x_len = x_len[0][0]
                x_len = x_len.item()
            len_list.append(x_len)

        tensor_len = torch.LongTensor(len_list)
        sorted_len, idx = tensor_len.sort(dim=0, descending=True)
        sorted_inputs = inputs[idx]
        _, reversed_idx =idx.sort(dim=0, descending=False)

        embeds = self.embedding(sorted_inputs)

        packed_embed = pack_padded_sequence(embeds, sorted_len, batch_first=True, enforce_sorted=True)
        packed_outputsRnn, _ = self.rnn(packed_embed)
        outputsRnn, _ = pad_packed_sequence(packed_outputsRnn, batch_first=True, total_length=inputs.shape[1])
        outputsRnn = outputsRnn[reversed_idx]

        outputsFc = self.mapping_layer(outputsRnn)
        output = torch.mean(outputsFc, dim=1)

        return output



def vectorize_dataset(dataset, char_to_idx, class_to_idx):
    vectorized_dataset = list()
    for word, lang in dataset:
        label = class_to_idx[lang]
        vectorized_word = list()
        for char in word:
            vectorized_word.append(char_to_idx.get(char, 1))  # Get the char index otherwise set to unknown char
        vectorized_dataset.append((vectorized_word, label))
    return vectorized_dataset


def load_data(filename):
    examples = list()
    with open(filename, encoding='utf-8') as fhandle:
        for line in fhandle:
            examples.append(line[:-1].split())
    return examples


def create_indexes(examples):
    char_to_idx = {"<pad>": 0, "<unk>": 1}
    class_to_idx = {}

    for word, lang in examples:
        if lang not in class_to_idx:
            class_to_idx[lang] = len(class_to_idx)
        for char in word:
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)
    return char_to_idx, class_to_idx


def make_max_padded_dataset(dataset):
    max_length = max([len(w) for w, l in dataset])
    tensor_dataset = torch.zeros((len(dataset), max_length), dtype=torch.long)
    labels = list()
    for i, (word, label) in enumerate(dataset):
        tensor_dataset[i, :len(word)] = torch.LongTensor(word)
        labels.append(label)
    return tensor_dataset, torch.LongTensor(labels)


def collate_examples(samples):
    # TODO: Cette fonction devrait faire du "padding on batch"
    # i.e. utiliser la longueur de la séquence la plus longue
    # de la batch et non pas du jeu de données complet.
    max_length = max([len(w) for w, l in samples])
    tensor_samples = torch.zeros((len(samples), max_length), dtype=torch.long)
    labels = list()
    for i, (word, label) in enumerate(samples):
        tensor_samples[i, :len(word)] = torch.LongTensor(word)
        labels.append(label)
    return tensor_samples, torch.LongTensor(labels)


def main():
    batch_size = 128
    training_set = load_data("./data-q2/train.txt")
    test_set = load_data("./data-q2/test.txt")

    char_to_idx, class_to_idx = create_indexes(training_set)

    vectorized_train = vectorize_dataset(training_set, char_to_idx, class_to_idx)
    vectorized_test = vectorize_dataset(test_set, char_to_idx, class_to_idx)

    X_train, y_train = make_max_padded_dataset(vectorized_train)
    X_test, y_test = make_max_padded_dataset(vectorized_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 1: Créez un réseau simple qui prend en entré des exemples de longueur fixes (max length)
    network = WordClassifier(char_to_idx, 10, 10, len(class_to_idx))
    optim = torch.optim.SGD(network.parameters(), lr=0.8, momentum=0.8)
    model = Model(network, optim, 'cross_entropy', batch_metrics=['accuracy'])
    scheduler = pt.ReduceLROnPlateau(monitor="loss", factor=0.01, patience=0, threshold=1e-1, verbose=True)
    model.fit_generator(train_loader, epochs=5, callbacks=[scheduler])
    loss, acc = model.evaluate_generator(test_loader)
    logging.info("1 - Loss: {}\tAcc:{}".format(loss, acc))

    # # 2: Faites en sorte que le padding soit fait "on batch"
    # # Le tout devrait se passer dans la fonction collate_examples
    train_loader = DataLoader(vectorized_train, batch_size=128, shuffle=True, collate_fn=collate_examples)
    test_loader = DataLoader(vectorized_test, batch_size=128, collate_fn=collate_examples)
    optim = torch.optim.SGD(network.parameters(), lr=0.8, momentum=0.8)
    model = Model(network, optim, 'cross_entropy', batch_metrics=['accuracy'])
    scheduler = pt.ReduceLROnPlateau(monitor="loss", factor=0.01, patience=0, threshold=1e-1, verbose=True)
    model.fit_generator(train_loader, epochs=5, callbacks=[scheduler])
    loss, acc = model.evaluate_generator(test_loader)
    logging.info("2 - Loss: {}\tAcc:{}".format(loss, acc))

    # # 3: Créez une architecture qui gère convenablement des séquences de longueur différentes
    network = WordClassifierHandlingPadding(char_to_idx, 10, 10, len(class_to_idx))
    optim = torch.optim.SGD(network.parameters(), lr=0.8, momentum=0.8)
    model = Model(network, optim, 'cross_entropy', batch_metrics=['accuracy'])
    scheduler = pt.ReduceLROnPlateau(monitor="loss", factor=0.01, patience=0, threshold=1e-1, verbose=True)
    model.fit_generator(train_loader, epochs=5, callbacks=[scheduler])
    loss, acc = model.evaluate_generator(test_loader)
    logging.info("3 - Loss: {}\tAcc:{}".format(loss, acc))



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()
