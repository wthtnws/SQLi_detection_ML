import re
from array import *
import numpy as np


def clean_query(target_string):
    res = re.sub("`", "", target_string)
    res = re.sub("/\*\*/", "", res)
    res = re.sub("\( +\)", "", res)

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
    res = re.sub("а-яА-Я", "", res)
    return res.strip()


def degree_centrality(adj_matrix, tok_str):
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

    return degree_list


FILE_TOKENS = []
FILE_REGEX = []


def generate_csv():
    with open("D:\\Study\\LABS\\NIR\\SQLi_detection_ML\\Data\\TOKENIZEME") as target_strings:
        for target in target_strings:

            target = clean_query(target).lower().strip("\n")
            print("BEFORE------> " + target)

            for i in range(len(FILE_REGEX)):
                reg = FILE_REGEX[i].strip("\n")
                tok = FILE_TOKENS[i].strip("\n")
                target = re.sub(reg, tok, target)

            target = re.sub("(CHR|STR) +DOT +(STR|STAR)", "USRCOL", target)
            target = re.sub("EXEC +STR", "EXEC USRPROC", target)
            target = re.sub("ATR +STR", "_STR_", target)

            tok_list = target.split(" ")
            replace_str = ""
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
                elif i == "AS":
                    replace_str = "_STR_"
                elif i == "VALUE":
                    replace_str = "_STR_"
                elif i == "USRCOL":
                    replace_str = "_STR_"
                elif i == "STR":
                    target = re.sub("\\bSTR\\b", replace_str, target, 1)

            target = re.sub("\\b_STR_\\b", "STR", target)
            print("AFTER------> " + target + "\n")
            adj_matrix = build_adj_matrix(target)
            degree_list = degree_centrality(adj_matrix, target)

            ###########################


def build_adj_matrix(s):
    tok_list = s.split(" ")
    N = len(tok_list)
    unique_tok_list = list(set(tok_list))
    unique_tok_list.sort()
    n = len(unique_tok_list)

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
            adj_matrix[ii][jj] += i + windw_size - j
            adj_matrix[jj][ii] = adj_matrix[ii][jj]

    return adj_matrix


if __name__ == '__main__':
    file = open("D:\\Study\\LABS\\NIR\\SQLi_detection_ML\\Data\\DictionaryTokens", "r")
    FILE_TOKENS = file.readlines()
    file.close()
    file = open("D:\\Study\\LABS\\NIR\\SQLi_detection_ML\\Data\\DictionaryRegExpressions", "r")
    FILE_REGEX = file.readlines()
    file.close()
    generate_csv()

    # file.close()
