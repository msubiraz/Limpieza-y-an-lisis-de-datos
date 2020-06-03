import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from scipy.stats import shapiro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pylab as py 

# Cargamos el fichero con los datos de entranamiento
train = pd.read_csv("train.csv")

# Mostramos una previsualización de los datos y un resumen de las distintas variables
print(train.head())
train.describe()


#4. Análisis de los datos. 
#4.1. Selección de los grupos de datos que se quieren analizar/comparar (planificación de los análisis a aplicar). 





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
        print(col, 'sigue una distribución normal (rechazamos la H0)')
    else:
        print('No podemos asumir normalidad para ',col, '(rechazamos la H0)')
        
        
        
        
# Conclusión: Despues de analizar la normalidad de las distintas variables, solo podemos assumir normalidad
#para la variable Age.



#4.3. Aplicación de pruebas estadísticas para comparar los grupos de datos. En función de los datos y el objetivo del estudio, aplicar pruebas de contraste de hipótesis, 
#correlaciones, regresiones, etc. Aplicar al menos tres métodos de análisis diferentes. 


  




