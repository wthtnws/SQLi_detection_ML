import re
import csv
from typing import Dict

import networkx as nx
import pandas as pd
from stellargraph import StellarGraph
import matplotlib
from array import *
import numpy as np
from stellargraph.layer import GCNSupervisedGraphClassification
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

FILE_TOKENS = []
FILE_REGEX = []
SORT_TOKENS = []
MAIN_PATH = "D:\\Study\\Polytech\\NIR\\SQLi_detection_ML\\Data\\"


def clean_query(target_string):
    for symb in target_string:
        if symb.isascii() == 0:
            target_string = target_string.replace(symb, "")

    res = re.sub("`", "", target_string)
    res = re.sub("/\*\*/", " ", res)
    res = re.sub("\( +\)", " ", res)

    res = re.sub("\.", " . ", res)
    res = re.sub(",", " , ", res)
    res = re.sub("&", " & ", res)
    res = re.sub("\|", " | ", res)
    res = re.sub("~", " ~ ", res)
    res = re.sub("&", " & ", res)
    res = re.sub("!", " ! ", res)
    res = re.sub("#", " # ", res)
    res = re.sub("\$", " $ ", res)
    res = re.sub("%", " % ", res)
    res = re.sub("\^", " ^ ", res)
    res = re.sub("!", " ! ", res)
    res = re.sub("\*", " * ", res)
    res = re.sub("-", " - ", res)
    res = re.sub("\+", " + ", res)
    res = re.sub("=", " = ", res)
    res = re.sub("\(", " ( ", res)
    res = re.sub("\)", " ) ", res)
    res = re.sub("\{", " { ", res)
    res = re.sub("}", " } ", res)
    res = re.sub("\[", " [ ", res)
    res = re.sub("]", " ] ", res)
    res = re.sub(r"\\", " \ ", res)
    res = re.sub(":", " : ", res)
    res = re.sub(";", " ; ", res)
    res = re.sub("\'", " \' ", res)
    res = re.sub("\"", " \" ", res)
    res = re.sub("<", " < ", res)
    res = re.sub(">", " > ", res)
    res = re.sub("\?", " ? ", res)
    res = re.sub("/", " / ", res)
    res = re.sub("@", " @ ", res)
    res = re.sub("’", " ’ ", res)

    res = re.sub("  +", " ", res)

    res = re.sub("& &", "&&", res)
    res = re.sub("\| \|", "||", res)
    res = re.sub("! =", "!=", res)
    res = re.sub("< >", "<>", res)
    res = re.sub("- -", "--", res)

    res = re.sub("\`|\/\*\*\/|\( *\)", "", res)
    return res.strip()


def tokenize_query(target):
    print("BEFORE------> " + target)
    target = clean_query(target).lower().strip("\n")

    for i in range(len(FILE_REGEX)):
        reg = FILE_REGEX[i]
        tok = FILE_TOKENS[i]
        target = re.sub(reg, tok, target)

    target = re.sub("(CHR|STR) +DOT +(STR|STAR)", "USRCOL", target)
    target = re.sub("EXEC +STR", "EXEC USRPROC", target)
    target = re.sub("ATR +STR", "_STR_", target)

    tok_list = target.split(" ")
    replace_str = "_STR_"
    for i in tok_list:
        if i == "SELECT":
            replace_str = "USRCOL"
        elif i == "FROM":
            replace_str = "USRTBL"
        elif i == ("INSERT"):
            replace_str = "USRTBL"
        elif i == "UPDATE":
            replace_str = "USRTBL"
        elif i == "JOIN":
            replace_str = "USRTBL"
        elif i == "WHERE":
            replace_str = "USRCOL"
        elif i == "ORDERBY":
            replace_str = "USRCOL"
        elif i == "SELECTTOP":
            replace_str = "USRCOL"
        elif i == "AS":
            replace_str = "_STR_"
        elif i == "VALUE":
            replace_str = "_STR_"
        elif i == "USRCOL":
            replace_str = "_STR_"
        elif i == "STR":
            target = re.sub("\\bSTR\\b", replace_str, target, 1)

    target = re.sub("\\b_STR_\\b", "STR", target)
    print("AFTER------> " + target + "\n\n")

    return target


def add_word(container: Dict[str, int], word: str):
    count = container.get(word, -1)
    container[word] = count + 1
    return container[word]


def generate_unique_IDs(tok_list):
    tok_count_encounter = {}
    tok_IDs = []
    print (len(tok_list))
    for i in range(len(tok_list)):
        count = add_word(tok_count_encounter, tok_list[i])
        tok_IDs.append(tok_list[i] + "_" + str(count + 1))
        #print(tok_IDs[i])

    return tok_IDs


def generate_edges_data(tok_IDs):
    N = len(tok_IDs)

    windw_size = 5

    source_for_edge = []
    target_for_edge = []
    weight_for_edge = []

    for src_tok_i in range(N):
        if src_tok_i + windw_size <= N:
            p = src_tok_i + windw_size
        else:
            p = N

        for trgt_tok_i in range(src_tok_i + 1, p):
            source_for_edge.append(tok_IDs[src_tok_i])
            target_for_edge.append(tok_IDs[trgt_tok_i])

            new_weight = src_tok_i + windw_size - trgt_tok_i
            weight_for_edge.append(new_weight)

    edges_data = pd.DataFrame(
        {
            "source": source_for_edge,
            "target": target_for_edge,
            "weight": weight_for_edge,
        }
    )

    return edges_data


def calculate_position_encoding(seq_len, d, extra_feature_N=1, n=10000):
    P = np.zeros((seq_len, d + extra_feature_N))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


def generate_columns_name(pos_feature_N):
    columns_name = []

    for i in range(pos_feature_N):
        columns_name.append("posenc_" + str(++i))

    columns_name.append("tok_id")

    return columns_name


def generate_nodes_data(tok_list, tok_IDs, pos_feature_N: int = 4):
    N = len(tok_list)

    node_features = calculate_position_encoding(N, pos_feature_N)  # feature_N is always even

    for i in range(N):
        sort_tok_i = SORT_TOKENS.index(tok_list[i]) + 1
        node_features[
            i, pos_feature_N] = sort_tok_i  # добавляем номер соответствующего токена в отсортированном массива
        #print(tok_IDs[i], sort_tok_i)

    columns_name = generate_columns_name(pos_feature_N)

    nodes_data = pd.DataFrame(
        node_features,
        columns=columns_name,
        index=tok_IDs
    )

    return nodes_data


def build_stellar_graph(tok_query):
    tok_list = tok_query.split(" ")
    N = len(tok_list)

    tok_IDs = generate_unique_IDs(tok_list)

    edges = generate_edges_data(tok_IDs)

    nodes = generate_nodes_data(tok_list, tok_IDs)

    stellarG = StellarGraph(
        {"token": nodes}, {"bond": edges}
    )
    print(stellarG.info())
    #print(stellarG.edges())
    #print(stellarG.nodes())
    print(stellarG.node_features())

    return stellarG

#TODO: сделать генерацию лейблов + соединить датасеты
def generate_graph_labels():
    zeros_labels = [0 for x in range(0, 132)]
    zeros_labels.extend([1 for x in range(0, 278)])
    graphs_labels = pd.DataFrame({'label': zeros_labels})

    print(graphs_labels.value_counts().to_frame())


def generate_stellar_dataset(input_filename: str):
    with open(input_filename, encoding="UTF-8") as target_file:
        graphs = []

        for target in target_file:
            tokenized = tokenize_query(target)

            stellar_graph = build_stellar_graph(tokenized)

            graphs.append(stellar_graph)

        return graphs

#TODO: реализовать модель
def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model

if __name__ == '__main__':

    file = open(MAIN_PATH + "DictionaryTokens", "r")
    FILE_TOKENS = file.readlines()
    SORT_TOKENS = FILE_TOKENS.copy()
    SORT_TOKENS.sort()
    file.close()
    file = open(MAIN_PATH + "DictionaryRegExpressions", "r")
    FILE_REGEX = file.readlines()
    file.close()

    for i in range(len(FILE_TOKENS)):
        SORT_TOKENS[i] = SORT_TOKENS[i].strip("\n")
        FILE_TOKENS[i] = FILE_TOKENS[i].strip("\n")
        if i < len(FILE_REGEX):
            FILE_REGEX[i] = FILE_REGEX[i].strip("\n")

    inj_file = MAIN_PATH + "InjQueries.txt"
    benign_file = MAIN_PATH + "BenignQueries.txt"

    tokenizeme = MAIN_PATH + "TOKENIZEME"

    dataset = generate_stellar_dataset(tokenizeme)
