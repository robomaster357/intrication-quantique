import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression


######## affiche un histogramme
def affichagetab(data):
    obs=[]
    for i in range(len(data)):
        d=data[i]
        assert all(d.columns == ['alpha', 'X', 'Y']), 'Vérifier les noms des colonnes.'
        obs.append(d.alpha[d.X == 1]) # Observations : alpha tels que X == 1 (photon détecté)

    _ = plt.hist(obs, bins=180)   


def affichage(data):
    assert all(data.columns == ['alpha', 'X', 'Y']), 'Vérifier les noms des colonnes.'
    obs = data.alpha[(data.X == 1) & (data.Y==0)] # Observations : alpha tels que X == 1 (photon détecté).
    print(obs)
    #obs = data.alpha[data.X == 1]
    _ = plt.hist(obs, bins=180) 

def affichage3(data):
    assert all(data.columns == ['angle', 'beta', 'Xa', 'Ya', 'Xb', 'Yb']), 'Vérifier les noms des colonnes.'
    obs = data.angle[data.X == 1] # Observations : alpha tels que X == 1 (photon détecté).
    print(obs)
    #obs = data.alpha[data.X == 1]
    _ = plt.hist(obs, bins=180) 



######### Régression linéaire
def regression(data):
    # Comptage des observations dans des intervalles de 5°.
    obs = data.alpha[(data.X == 1) & (data.Y == 0)].to_numpy()
    (comptage, alpha_bords) = np.histogram(obs, bins = 36)

    # Calcul de cos²(α), où α est pris au centre des intervalles.
    # Conversion en tableau 2D pour les besoins de l'estimateur.
    alpha_centres = 0.5 * (alpha_bords[0:-1] + alpha_bords[1:])
    cos2_alpha = (np.cos(np.radians(alpha_centres)) ** 2).reshape(-1,1)
    # On ajoute une constante pour la regression
    cos2_alphaReg = sm.add_constant(cos2_alpha)

    # Régression linéaire.
    reg = sm.OLS(comptage, cos2_alphaReg).fit()

    # Tracé des comptages et de la régression trouvée.
    plt.scatter(cos2_alpha, comptage, label = 'Comptage')
    plt.plot(
        cos2_alpha,
        reg.predict(cos2_alphaReg),
        linewidth = 3,
        color = 'tab:orange',
        label = 'Régression'
    )
    plt.legend()

    # Affichage des scores.
    reg.summary()

def main():
    # Préchargement des jeux de données.
    df_1photon_x = pd.read_csv('jour1-1-facile/1photon-polar-x-alea.csv', sep=';')
    df_1photon_y = pd.read_csv('jour1-1-facile/1photon-polar-y-alea.csv', sep=';')
    df_1photon_45 = pd.read_csv('jour1-1-facile/1photon-polar-45-alea.csv', sep=';')

    df_1photon_x_bruit=pd.read_csv('jour1-2-bruit/1photon-polar-x-bruit-1e+1.csv', sep=';')
    df_1photon_45_bruit=pd.read_csv('jour1-2-bruit/1photon-polar-45-bruit-1e+1.csv', sep=';')

    df_1photon_d01=pd.read_csv('jour1-3-divers/jour1-divers-01.csv', sep=';')
    df_1photon_d02=pd.read_csv('jour1-3-divers/jour1-divers-02.csv', sep=';')
    df_1photon_d03=pd.read_csv('jour1-3-divers/jour1-divers-03-xx.csv', sep=';')
    df_1photon_d04=pd.read_csv('jour1-3-divers/jour1-divers-04-polar-circulaire.csv', sep=';')
    df_1photon_d05=pd.read_csv('jour1-3-divers/jour1-divers-05.csv', sep=';')
    df_1photon_d06=pd.read_csv('jour1-3-divers/jour1-divers-06.csv', sep=';')
    df_1photon_d07=pd.read_csv('jour1-3-divers/jour1-divers-07.csv', sep=';')
    df_1photon_d08=pd.read_csv('jour1-3-divers/jour1-divers-08.csv', sep=';')
    df_1photon_d09=pd.read_csv('jour1-3-divers/jour1-divers-09.csv', sep=';')

    #data=[df_1photon_x, df_1photon_y, df_1photon_45, df_1photon_x_bruit, df_1photon_45_bruit]
    data=pd.read_csv('polar-1photon-yunw9b.csv', sep=';')
    print(data)
    #print(df_1photon_d04)
    #data=df_1photon_d04
    affichage(data)
    plt.show()
    
main()