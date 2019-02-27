import os

data = "data"

with open(os.path.join(data, "cn_vocab.txt"), 'r', encoding='utf8') as fp:
    lines = fp.readlines()
    print(len(lines))
