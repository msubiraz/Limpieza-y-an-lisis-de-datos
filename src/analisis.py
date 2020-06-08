import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Cargamos el fichero con los datos de entranamiento
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Mostramos una previsualización de los datos y un resumen de las distintas variables
print(train.head())
train.describe()


# Creamos un array vacio para almacenar las edades que vamos a completar basados en pclass x gender

guess_ages = np.zeros((2,3))

# Ahora iteramos en la columna genero y clase para calcular las edades que falta

for dataset in combine:  
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_mean = guess_df.mean()
            age_std = guess_df.std()
            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            #age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

#4. Análisis de los datos. 
#4.1. Selección de los grupos de datos que se quieren analizar/comparar (planificación de los análisis a aplicar).

# Las principales hipótesis que queremos validar son: 
    #Gender -- Las mujeres tiene mejor ratio de supervivencia que los hombres
    #Age -- Los niños tiene preferencia sobre los adultos.
    #Passenger Class -- Los passajeros de 1a clase son los primeros en abandonar el barco en botes.
    #Fare -- Los pasajeros con coste del billete mayor, estan por delante de los pasajeros de su misma clase.
    #Solos o en group -- Los grupos icentivan al resto a buscarles, mientras que los pasajeros que viajan solos tiene mas
    #                   numeros a ser olvidados.
#Factores irrelevantes
    #Cabin #, Ticket # -- No tiene relevancia para la predicción
    #Name -- we don't have survival data for all passengers
#Posibles variables determiantes
    #Título -- Miss, Master, Mr, Mrs, otros.
    #Puerto de embarque
    #Niños vs adultos

# La variable campo la renombramos a título 
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    #convert female / male to 1, 0
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    #Convert titles to numeric categories -- helps w/ sklearn algorithms
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
#Eliminamos passangerId y Name de train, y name de test
train = train.drop(['Name','PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)

import seaborn as sns
from scipy.stats import norm
from scipy.stats import shapiro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pylab as py 
  
#4.2. Comprobación de la normalidad y homogeneidad de la varianza. 


# Generamos gráficos con las distribuciones de las variables
fig, ax = plt.subplots(figsize = (120, 100))
for i, col in enumerate(train.select_dtypes(include=np.number).columns):
    plt.subplot(25, 40, i+1)
    sns.distplot(train.select_dtypes(include=np.number).iloc[:,i],fit=norm, kde=False,ax=plt.gca())
    
plt.tight_layout()


# Los gráficos Q-Q ("Q" viene de cuantil) es un método gráfico para comparar la distribución de nuestra muestra contra una
# distribución normal teórica.
# Vamos a generar ahora los gráficos Q-Q:

for i, col in enumerate(train.select_dtypes(include=np.number).columns):
    sm.qqplot(train[col], line ='s') 
    py.show()


# Aplicamos ahora el test Shapiro-Wilk y validamos los resultados con un alpha de 0.05
#p <= alpha: rechazamos H0, no podemos asumir normalidad.
#p > alpha: No podemos rechazar H0, normalidad.


for i, col in enumerate(train.select_dtypes(include=np.number).columns):
    stat, p = shapiro(train[col])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpretar
    alpha = 0.05
    if p > alpha:
        print(col, 'sigue una distribución normal (aceptamos la H0)')
    else:
        print('No podemos asumir normalidad para ',col, '(rechazamos la H0)')
        
        
        
        
# Conclusión: Despues de analizar la normalidad de las distintas variables, solo podemos assumir normalidad
#para la variable Age.



#Analizamos ahora la homogeneidad de la varianza para para los supervivientes vs los fallecidos. Para ello, utilizaremos el Test de Levene. Podriamos utilizar el 
#F-test para la variable, ya que hemos assumido normalidad en su distribución.

from scipy.stats import levene

#separamos la muestra ne dos poblaciones segun la variable survided:
train.fillna(30, inplace = True)

train_1= train.loc[train['Survived'] == 1]
train_0= train.loc[train['Survived'] == 0]


for i, col in enumerate(train.select_dtypes(include=np.number).columns):
    stat, p = levene(train_1[col],train_0[col])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpretar
    alpha = 0.05
    if p > alpha:
        print(col, ' asume igualdad de varianzas (aceptamos la H0)')
    else:
        print('diferencia entre las varianzas muestrales para la variable ',col, '(rechazamos la H0)')
        
    
# Observamos que solo para la variable SibSp no podemos asumir igualdad de varianzas.

import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statistics
import scipy.stats as stats
from statsmodels.stats import proportion as proptests

#4.3. Aplicación de pruebas estadísticas para comparar los grupos de datos. En función de los datos y el objetivo del estudio, 
#aplicar pruebas de contraste de hipótesis, correlaciones, regresiones, etc. Aplicar al menos tres métodos de análisis 
#diferentes. 

#1. Mapa de correlaciones
plt.figure(figsize=(10,7))
sns.heatmap(train.corr(),annot=True,fmt=".2f")
# Vemos que el gráfico muestra que la mayor correlación la encontramos entre las variables Pclass y fare, como es intutivo 
# pensar que a mayor precio del bilete, mejor la classe.

#Si nos fijamos en las correlaciones con la clase Survived, vemos que los fatores mas correlacionados son el genero y el título 
#de la persona, que entre otros cosas refleja datos del género y la edad.

# PAra las variables categoricas, utilizamos la funciona get_dumies para crear columnas para cada unos de los valores 
#categoricos.
train = pd.get_dummies(train, drop_first = True)


#2. Estad´sitico de contraste
#Queremos analizar ahora el ratio de supervivencia por género, i determinar a partir de un estadístico de contraste si 
#la diferencia es significativa (ya esperamos como hemos indicado en nuestras hipótesis, que le ratio de supervivencia
#de las mujeres es superior al de los hombres).

mean_survived_male = train.loc[train['Sex'] == 0]['Survived'].astype(int).mean()
mean_survived_female = train.loc[train['Sex'] == 1]['Survived'].astype(int).mean()

print("Ratio de supervivencia de hombres: {mean_survived_male}".format(mean_survived_male = mean_survived_male))
print("Ratio de supervivencia de mujeres: {mean_survived_female}".format(mean_survived_female = mean_survived_female))


#Preparamos dos datasens con la variable survice para cada uno de los generos:
male = train[train['Sex'] == 0]['Survived']
female = train[train['Sex'] == 1]['Survived']

# Calculamos las medias
male_mean = male.mean()
female_mean = female.mean()

# Calculamos la varianza
male_var = statistics.variance(male)
female_var = statistics.variance(female)

# calculamos el estadístico de contraste
t = (female_mean - male_mean)/math.sqrt(female_var/len(female) + male_var/len(male))
print('T-value is {t}'.format(t = t))

# Fijamos los grados de libertad
df = len(male) + len(female) - 2

# Calculamos el valor-p
p = 1 - stats.t.cdf(t,df=train)
print('P-value is {p}'.format(p = p))

#Vemos que el valor del estadtístico T es mayor que el valor crítico t, por lo que concluimos que hay una 
#diferencia estadística significativa entre las dos poblaciones. Esto confira nuestra hipótesis inicial de que las mujeres
#tiene mayor probabilidades de sobrevivir.

#3. Logistic Regression
#Regresiones logísticas nos permiten crear un modelo para predecir el resultado de una variable categórica.

# Separamos los datos de train en dos dubconjuntos para entrenar y testear el modelo
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

#Creamos el modelo de regresión logístico y lo entrenamos con la particion de entrenamiento
logmodel = LogisticRegression(solver = 'liblinear')
logmodel.fit(X_train,y_train)

# Guradamos las predicciones y calculamos el accuracy de nuestro modelo

predictions = logmodel.predict(X_test)
predictions


print(classification_report(y_test,predictions))


#5. Representación de los resultados a partir de tablas y gráficas. 

#vamos a visualizar primero los ratios de supervivencia para cada una de las variables:
train.hist(color='g')
train[train["Survived"] == 1].hist()
plt.show()

# % Sobrevivientes por grupos
plt.figure(figsize=(14,10))
plt.suptitle("Percentage of survivors per group", fontsize=18)
plt.subplot(231)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(232)
sns.barplot('Age', 'Survived', data=train)
plt.subplot(233)
sns.barplot('Fare', 'Survived', data=train)
plt.subplot(234)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(235)
sns.barplot('Pclass', 'Survived', data=train)
plt.show()


