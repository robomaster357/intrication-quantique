################################ Importation des librairies ###############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression









################################ Affichage de l'histogramme du nombre de clics ###############################
def affichage_hist(data):
    assert all(data.columns == ['alpha', 'beta', 'Xa', 'Ya', 'Xb', 'Yb']), \
        'Vérifier les noms des colonnes.'

    # ✅ Événements valides : un seul clic de chaque côté
    valid = (
        ((data.Xa + data.Ya) == 1) &  # côté A : un seul détecteur
        ((data.Xb + data.Yb) == 1)    # côté B : un seul détecteur
    )

    data_valid = data[valid]

    # Observations filtrées
    alpha_obs = data_valid.alpha[data_valid.Xa == 1]
    beta_obs  = data_valid.beta[data_valid.Xb == 1]

    plt.hist(alpha_obs, bins=36, label='Trajectoire A (alpha)', color = "#728DB9")
    plt.hist(beta_obs, bins=36, label='Trajectoire B (beta)', color = "#F1A87E")
    plt.legend()
    plt.xlabel('Angles alpha/beta en degrés')
    plt.ylabel('Nombre de clics')
    plt.title('Événements avec détection simple de chaque côté')


"""def affichage_hist(data):
    assert all(data.columns == ['alpha', 'beta', 'Xa', 'Ya', 'Xb', 'Yb'])

    alpha_obs = data.alpha[(data.Xa == 1) & (data.Ya == 0)].to_numpy()
    beta_obs  = data.beta[(data.Xb == 1) & (data.Yb == 0)].to_numpy()

    # Bins fixes de 0 à 180 degrés
    bins = np.linspace(0, 180, 181)

    counts_alpha, _ = np.histogram(alpha_obs, bins=bins)
    counts_beta,  _ = np.histogram(beta_obs,  bins=bins)

    # Largeur d’un bin = 1°, on prend la moitié
    width = 0.5

    # Positions de départ des bins
    positions = bins[:-1]

    # Alpha à gauche, Beta à droite
    plt.bar(positions, counts_alpha, width=width, align='edge', label='Alpha')
    plt.bar(positions + width, counts_beta, width=width, align='edge', label='Beta')

    plt.legend()
    plt.xlabel('Angle (°)')
    plt.ylabel('Nombre de clics')
    plt.title('Histogrammes alternés alpha / beta')
    plt.show()
"""





################################ Calculs de la régression linéaire ###############################
"""def err_mod(obs, theta):
    # Comptage des observations dans des intervalles de 5°.
    #obs = data.alpha[(data.X == 1) & (data.Y == 0)].to_numpy()
    (comptage, alpha_bords) = np.histogram(obs, bins = 36)

    # Calcul de cos²(α - theta), où α est pris au centre des intervalles.
    # Conversion en tableau 2D pour les besoins de l'estimateur.
    alpha_centres = 0.5 * (alpha_bords[0:-1] + alpha_bords[1:])
    cos2_alpha = (np.cos(np.radians(alpha_centres-theta)) ** 2).reshape(-1,1)
    # On ajoute une constante pour la regression
    cos2_alphaReg = sm.add_constant(cos2_alpha)

    reg = sm.OLS(comptage, cos2_alphaReg).fit()
    b,a = reg.params
    r2 = reg.rsquared

    return (b,a,r2)"""

def err_mod(obs, theta):
    comptage, alpha_bords = np.histogram(obs, bins=36)
    alpha_centres = 0.5 * (alpha_bords[:-1] + alpha_bords[1:])

    cos2 = (np.cos(np.radians(alpha_centres - theta))**2).reshape(-1,1)
    X = sm.add_constant(cos2)

    weights = 1 / np.maximum(comptage, 1)
    reg = sm.WLS(comptage, X, weights=weights).fit()

    chi2 = np.sum(((comptage - reg.predict())**2) / np.maximum(comptage,1))
    b, a = reg.params
    return b, a, chi2

def affichage_reg(data, alpha_b, alpha_a, beta_b, beta_a, alpha_theta, beta_theta):
    assert all(data.columns == ['alpha', 'beta', 'Xa', 'Ya', 'Xb', 'Yb']), 'Vérifier les noms des colonnes.'
    alpha_obs = data.alpha[(data.Xa == 1) & (data.Ya == 0)] # Observations : alpha tels que X == 1 et Y == 0(photon détecté).
    beta_obs = data.beta[(data.Xb == 1) & (data.Yb == 0)]    
    (alpha_comptage, bords) = np.histogram(alpha_obs, bins = 36)
    (beta_comptage, _) = np.histogram(beta_obs, bins = 36)

    # Calcul de cos²(α - theta), où α est pris au centre des intervalles.
    # Conversion en tableau 2D pour les besoins de l'estimateur.
    centres = 0.5 * (bords[0:-1] + bords[1:])
    cos2_alpha = (np.cos(np.radians(centres-alpha_theta)) ** 2).reshape(-1,1)
    cos2_beta = (np.cos(np.radians(centres-beta_theta)) ** 2).reshape(-1,1)
    # On ajoute une constante pour la regression
    cos2_alphaReg = sm.add_constant(cos2_alpha)
    cos2_betaReg = sm.add_constant(cos2_beta)

    # Tracé des comptages et de la régression trouvée.
    plt.scatter(cos2_alpha, alpha_comptage, label = f'Comptage en fonction de alpha pour theta = {alpha_theta}')
    plt.scatter(cos2_beta, beta_comptage, label = f'Comptage en fonction de beta pour theta = {beta_theta}')

    #Tracé de la régression linéaire
    Xideal = np.array([0,1]).reshape(-1, 1)
    # On ajoute une constante pour la regression
    XidealReg = sm.add_constant(Xideal)
    plt.plot(Xideal, np.dot(XidealReg,[alpha_b, alpha_a]), label = 'Régression linéaire pour alpha')
    plt.plot(Xideal, np.dot(XidealReg,[beta_b, beta_a]), label = 'Régression linéaire pour beta')

    plt.legend()


def regression(data):
    assert all(data.columns == ['alpha', 'beta', 'Xa', 'Ya', 'Xb', 'Yb']), 'Vérifier les noms des colonnes.'
    alpha_obs = data.alpha[(data.Xa == 1) & (data.Ya == 0)].to_numpy()
    beta_obs = data.beta[(data.Xb == 1) & (data.Yb == 0)].to_numpy()

    #Calculs de la meilleure régression linéaire
    alpha_reg_a, alpha_reg_b, alpha_best, alpha_theta = 0,0,100,0
    beta_reg_a, beta_reg_b, beta_best, beta_theta = 0,0,100,0
    for theta in range(0,191):
        alpha_b, alpha_a, alpha_chi2 = err_mod(alpha_obs, theta)
        beta_b, beta_a, beta_chi2 = err_mod(beta_obs, theta)
        if alpha_chi2<alpha_best:
            alpha_best = alpha_chi2
            alpha_reg_a = alpha_a
            alpha_reg_b = alpha_b
            alpha_theta = theta
        if beta_chi2<beta_best:
            beta_best = beta_chi2
            beta_reg_a = beta_a
            beta_reg_b = beta_b
            beta_theta = theta
    
    #Rendu des modèles
    print(f"Trajectoire A (alpha): theta = {alpha_theta}, Comptage = {alpha_reg_a} * cos²(alpha-theta) + {alpha_reg_b}")
    print(f"Trajectoire B (beta): theta = {beta_theta}, Comptage = {beta_reg_a} * cos²(beta-theta) + {beta_reg_b}")
    affichage_reg(data, alpha_reg_b, alpha_reg_a, beta_reg_b, beta_reg_a, alpha_theta, beta_theta)









################################ Indépendance / Intrication des photons ###############################
def E_cal(data):
    valid = ((data.Xa + data.Ya) == 1) & ((data.Xb + data.Yb) == 1)
    d = data[valid]

    angles_a = sorted(d.alpha.unique())
    angles_b = sorted(d.beta.unique())

    E = {}

    for a in angles_a:
        for b in angles_b:
            subset = d[(d.alpha == a) & (d.beta == b)]
            N = len(subset)
            if N == 0:
                continue

            P_xx = np.sum((subset.Xa==1)&(subset.Xb==1)) / N
            P_yy = np.sum((subset.Ya==1)&(subset.Yb==1)) / N
            P_xy = np.sum((subset.Xa==1)&(subset.Yb==1)) / N
            P_yx = np.sum((subset.Ya==1)&(subset.Xb==1)) / N

            E[(a,b)] = P_xx + P_yy - P_xy - P_yx
    
    return E

def S(data):
    E=E_cal(data)

    angles_a = sorted(set(a for a, _ in E.keys()))
    angles_b = sorted(set(b for _, b in E.keys()))

    best_S = 0
    best_combo = None

    for a, a2 in itertools.permutations(angles_a, 2):
        for b, b2 in itertools.permutations(angles_b, 2):

            # Vérifier que toutes les valeurs existent
            if (a,b) in E and (a2,b) in E and (a,b2) in E and (a2,b2) in E:
                S = E[(a,b)] + E[(a2,b)] - E[(a,b2)] + E[(a2,b2)]
                
                if abs(S) > best_S:
                    best_S = abs(S)
                    best_combo = (a, a2, b, b2)

    print(f"S max = {best_S} obtenu pour les angles optimaux : {best_combo}")
    if best_S>2.7: print("Les photons sont intriqués")
    else: print("Les photons sont indépendants")
    return best_S


def main():
    #Chargement de tous les jeux de données
    df_45_1M=pd.read_csv('jour2-1-facile/2photons-45-1M.csv', sep=';')
    df_autre_1M=pd.read_csv('jour2-1-facile/2photons-autre-1M.csv', sep=';')
    df_circ_1M=pd.read_csv('jour2-1-facile/2photons-circ-1M.csv', sep=';')
    df_epr_1M=pd.read_csv('jour2-1-facile/2photons-epr-1M.csv', sep=';')
    df_xx_1M=pd.read_csv('jour2-1-facile/2photons-xx-1M.csv', sep=';')
    df_yy_1M=pd.read_csv('jour2-1-facile/2photons-yy-1M.csv', sep=';')

    df_epr_1M_moy=pd.read_csv('jour2-2-moyen/2photons-epr.csv', sep=';')
    df_xx_1M_moy=pd.read_csv('jour2-2-moyen/2photons-xx.csv', sep=';')
    df_yy_1M_moy=pd.read_csv('jour2-2-moyen/2photons-yy.csv', sep=';')

    df_8wdvld=pd.read_csv('quiz 2/polar-2photons-8wdvld.csv', sep=',')
    df_ig509d=pd.read_csv('quiz 2/polar-2photons-ig509d.csv', sep=',')
    df_ljkq13=pd.read_csv('quiz 2/polar-2photons-ljkq13.csv', sep=',')

    #Choix du fichier à analyser
    data=df_epr_1M_moy

    #Affichage du set du jeux de données sous forme de double histogrammes
    affichage_hist(data)
    S_value=S(data)
    plt.show()
    
    #Trouver les paramètres des schémas expériementaux par régression linéaire
    #regression(data)
    #plt.show()
    
main()