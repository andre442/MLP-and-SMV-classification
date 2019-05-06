"""
Created on Sat May  4 15:36:25 2019
Comparativo de classificadores SMV e Rede Multi Layer Perceptron
@author: Andre Iarozinski
"""
import pandas as pd
import seaborn as sns
base = pd.read_csv('heart.csv')
#base.head()

previsores = base.iloc[:, 0:13].values
classe = base.iloc[:, 13].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.3)

# utilizando classificador SVM
from sklearn.svm import SVC
classificador = SVC(kernel = 'rbf', random_state = 0, C = 2.0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#plotando matriz de confusão do algoritmo SVM
sns.heatmap(matriz,cmap='coolwarm', annot=True).set_xlabel("Matriz SVM")
# precisão svm = 0.824

#utilizando rede neural MLP
from sklearn.neural_network import MLPClassifier
classificadorR = MLPClassifier(verbose = True, max_iter=1000, tol=0.000010)
classificadorR.fit(previsores_treinamento, classe_treinamento)
previsoes = classificadorR.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisaoR = accuracy_score(classe_teste, previsoes)
matrizR = confusion_matrix(classe_teste, previsoes)

#plotando matriz de confusão do algoritmo MLP
sns.heatmap(matrizR,cmap='coolwarm', annot=True).set_xlabel("Matriz MPL")
# precisão da rede = 0.868






