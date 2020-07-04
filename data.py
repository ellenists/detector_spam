import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import _pickle as c

print(" [DETECTOR DE SPAM] Processamento de dados, treinamento e teste do modelo de ML\n\n")
def salvar_modelo(clf, nome):
    with open(nome, 'wb') as fp:
        c.dump(clf, fp)
    print(" Modelo salvo em arquivo.")


def construir_dicionario():
    dir = "emails/"
    arq = os.listdir(dir)
    print(" Lendo todos os arquivos de e-mail.\n")
    print(" Obtendo lista de todas as palavras presentes em todos os e-mails...\n") 
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
    print(" Removendo palavras com caracteres nao-alfabeticos...\n")
    print(" Criando um dicionario com as palavras restantes.\n")
    print(" 10 palavras mais comuns: ")
    print(dic.most_common(10))
    return dic.most_common(3000)

print("\n Criando dataset com os dados coletados...\n")
def criar_dataset(dic):
    dir = "emails/"
    arq = os.listdir(dir)
    emails = [dir + email for email in arq]
    feature_set = []
    labels = []
    c = len(emails)

    print("\n Adicionando as palavras obtidas ao feature set...\n")
    print(" Classificando cada palavra como ham ou spam - labels...\n")
    for email in emails:
        data = []
        eml = open(email, encoding="latin-1")
        palvrs = eml.read().split(' ')
        for entry in dic:
            data.append(palvrs.count(entry[0]))
        feature_set.append(data)

        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
        c = c - 1
    return feature_set, labels


d = construir_dicionario()
features, labels = criar_dataset(d)

print(" Dividindo nosso dataset entre treino e teste, 80% treino, 20% teste,")
print(" utilizando as features e labels de treino e de teste...\n")
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

print(" Criando o modelo de Machine Learning com base no dataset de treino...\n")
clf = MultinomialNB()
clf.fit(x_train, y_train)

print(" Realizando predicao para testar o modelo que acabamos de criar...\n")
pred = clf.predict(x_test)
print(" Pontuacao de acuracia atingida pelo nosso modelo: ", end = '') 
print(accuracy_score(y_test, pred))
print("\n Salvando modelo em forma de arquivo para uso posterior.\n")
salvar_modelo(clf, "text-classifier.mdl")
