import _pickle as c
import os
from sklearn import *
from collections import Counter

print(" Utilizando o modelo criado para classificar\n as seguintes mensagens entre Ham e Spam:")
def carregar_modelo(clf_file):
    with open(clf_file, 'rb') as fp:
        clf = c.load(fp)
    return clf

# Dicionario para classificar as palavras do input de modo que possam ser enviadas para o modelo de ML
def construir_dicionario():
    dir = "emails/"
    arq = os.listdir(dir)
    emails = [dir + email for email in arq]
    palvrs = []
    c = len(emails)
    for email in emails:
        eml = open(email, encoding="latin-1")
        blob = eml.read()
        palvrs += blob.split(" ")
        c -= 1

    for i in range(len(palvrs)):
        if not palvrs[i].isalpha():
            palvrs[i] = ""

    dic = Counter(palvrs)
    del dic[""]
    return dic.most_common(3000)


clf = carregar_modelo("text-classifier.mdl")
d = construir_dicionario()


while True:
    features = []
    inp = input("\n> ").split()
    if inp[0] == "fim":
        break
    for pal in d:
        features.append(inp.count(pal[0]))
    res = clf.predict([features])
    print([" Sem spam (:", " Spam!"][res[0]])