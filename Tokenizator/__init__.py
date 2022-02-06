import re


def add_whitespace(target_string):
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

    res = re.sub("  +", " ", res)

    res = re.sub("& &", "&&", res)
    res = re.sub("\| \|", "||", res)
    res = re.sub("! =", "!=", res)
    res = re.sub("< >", "<>", res)


    print (res)


def tokenize_string(target_string):
    res = re.sub(" (?i)and ", " ANDDDD ", target_string, )
    dictionary = {r" 0[xX][0-9a-fA-F]+ " : "HEX",
                  }
    print (res)


if __name__ == '__main__':
    target_string = "1 ' ) as qcse where 4147 = 4147 AND waitfor delay ' 0 : 0 : 5 ' - - "
    '''
    with open("D:\\Study\\LABS\\NIR\\SQLi_detection_ML\\Data\\tmpText") as file:
        for item in file:
            #print(item)
            add_whitespace(item)
            '''
    tokenize_string(target_string)
    # file.close()
