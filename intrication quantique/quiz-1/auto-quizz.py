import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression

data=pd.read_csv('polar-1photon-pg7py1.csv', sep=',')

######## Affichage de l'histogramme
assert all(data.columns == ['alpha', 'X', 'Y']), 'Vérifier les noms des colonnes.'
obs = data.alpha[data.X == 1] # Observations : alpha tels que X == 1 (photon détecté).
#obs = data.alpha[data.X == 1]
_ = plt.hist(obs, bins=180)
plt.show()