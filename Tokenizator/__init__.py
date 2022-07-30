import re
import csv
import networkx as nx
import matplotlib
from array import *
import numpy as np

FILE_TOKENS = []
FILE_REGEX = []
SORT_TOKENS = []
MAIN_PATH = "C:\\"


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


def build_graph(s):


    tok_list = s.split(" ")
    N = len(tok_list)
    unique_tok_list = list(set(tok_list))
    unique_tok_list.sort()
    n = len(unique_tok_list)

    G = nx.Graph()
    G.add_nodes_from(unique_tok_list)

    adj_matrix = [0] * n
    windw_size = 5
    for i in range(n):
        adj_matrix[i] = [0] * n

    for i in range(N):
        if i + windw_size <= N:
            p = i + windw_size
        else:
            p = N
        ii = unique_tok_list.index(tok_list[i])
        for j in range(i + 1, p):
            jj = unique_tok_list.index(tok_list[j])
            new_weight = adj_matrix[ii][jj] + i + windw_size - j
            adj_matrix[ii][jj] = new_weight
            adj_matrix[jj][ii] = new_weight
            G.add_edge(tok_list[i], tok_list[j], weight=new_weight)

    #print(adj_matrix)
    #print("\n")

    return G, adj_matrix


def calculate_centralities(G, adj_matrix, tok_str):
    tok_list = tok_str.split(" ")
    N = len(tok_list)
    unique_tok_list = list(set(tok_list))
    unique_tok_list.sort()
    n = len(unique_tok_list)

    degree_list = [0] * n
    for i in range(n):
        degree_list[i] += adj_matrix[i][i]
        for j in range(n):
            degree_list[i] += adj_matrix[i][j]

    for i in range(n):
        print(unique_tok_list[i] + " deg = ", degree_list[i])

    print(G.adj)
    closeness = nx.closeness_centrality(G, distance='weight')
    closeness_list = list(closeness.values())
    print(closeness)
    print(closeness_list)

    eigenvector = nx.eigenvector_centrality(G, 100, 1.0e-6, None, 'weight')
    eigenvector_list = list(eigenvector.values())
    print(eigenvector)
    print(eigenvector_list)
    # print(nx.katz_centrality(G = G,max_iter= 100, weight= 'weight',tol=1.0e-2, alpha=0.5))

    degree_nx = nx.degree_centrality(G)
    degree_list_nx = list(degree_nx.values())
    print(degree_nx)
    print(degree_list_nx)

    return degree_list, degree_list_nx, closeness_list, eigenvector_list

def normalize_vector(data_vector: list):
    normal = ["0"] * len(data_vector)
    tmp2 = data_vector.copy()
    tmp2.sort()
    max = tmp2.pop()
    for i in range(len(data_vector)):
        normal[i] = str(data_vector[i] / max)
    return normal


def generate_csv(input_filename: str, output_filelist: list, label: str):
    with open(input_filename, encoding="UTF-8") as target_file:

        for target in target_file:
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

            G, adj_matrix = build_graph(target)
            degree_list, \
            degree_list_nx, \
            closeness_list, eigenvector_list = calculate_centralities(G, adj_matrix, target)

            degree_vector = [0] * len(SORT_TOKENS)
            degree_nx_vector = [0] * len(SORT_TOKENS)
            closeness_vector = [0] * len(SORT_TOKENS)
            eigenvector_vector = [0] * len(SORT_TOKENS)

            unique_tok_list = list(set(target.split(" ")))
            unique_tok_list.sort()

            for i in range(len(degree_list)):
                vector_index = SORT_TOKENS.index(unique_tok_list[i])
                degree_vector[vector_index] = degree_list[i]
                degree_nx_vector[vector_index] = degree_list_nx[i]
                closeness_vector[vector_index] = closeness_list[i]
                eigenvector_vector[vector_index] = eigenvector_list[i]

            normalized_degree = normalize_vector(degree_vector).copy()

            normalized_degree.append(label)
            degree_nx_vector.append(label)
            closeness_vector.append(label)
            eigenvector_vector.append(label)

            vector_metrics_list = [normalized_degree,
                            degree_nx_vector,
                            closeness_vector,
                            eigenvector_vector]

            for i in range(0,4):

                file = open(output_filelist[i], encoding="UTF-8", mode="a+")
                file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
                file_writer.writerow(vector_metrics_list[i])
                file.close()

            '''
            print("SAMPLE STRING------> \n")
            for i in range(len(SORT_TOKENS)):
            print(SORT_TOKENS[i] +" = ",degree_vector[i])
            '''


if __name__ == '__main__':
    file = open(MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\DictionaryTokens", "r")
    FILE_TOKENS = file.readlines()
    SORT_TOKENS = FILE_TOKENS.copy()
    SORT_TOKENS.sort()
    file.close()
    file = open(MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\DictionaryRegExpressions", "r")
    FILE_REGEX = file.readlines()
    file.close()

    inj_file = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\InjQueries.txt"
    benign_file = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\BenignQueries.txt"
    csv_file = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\Centrality_Measure_Dataset.csv"

    tokenizeme = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\TOKENIZEME"
    tokenizeme_out_degree = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\DEGREE_CENTR.csv"
    tokenizeme_out_degree_nx = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\DEGREE_NX_CENTR.csv"
    tokenizeme_out_closeness = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\CLOSENESS_CENTR.csv"
    tokenizeme_out_eigenvector = MAIN_PATH + "LABS\\NIR\\SQLi_detection_ML\\Data\\EIGENVECTOR_CENTR.csv"

    output_filelist = [tokenizeme_out_degree,
                           tokenizeme_out_degree_nx,
                           tokenizeme_out_closeness,
                           tokenizeme_out_eigenvector]

    print(output_filelist)


    for i in range(len(FILE_TOKENS)):
        SORT_TOKENS[i] = SORT_TOKENS[i].strip("\n")
        FILE_TOKENS[i] = FILE_TOKENS[i].strip("\n")
        if i < len(FILE_REGEX):
            FILE_REGEX[i] = FILE_REGEX[i].strip("\n")

    for file_path in output_filelist:
        file = open(file_path, encoding="UTF-8", mode="w")
        file_writer = csv.writer(file, delimiter=",", lineterminator="\r")

        SORT_TOKENS.append("LABEL")
        file_writer.writerow(SORT_TOKENS)
        SORT_TOKENS.pop()
        file.close()

    #generate_csv(inj_file, csv_file, "1")
    #generate_csv(benign_file, csv_file, "0")
    generate_csv(tokenizeme, output_filelist, "0")

    # не забыть удалить повторяющиеся имена фичей, начинается с ADMIN...
