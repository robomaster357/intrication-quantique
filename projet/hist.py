import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def nb_clics(data):
    alpha_obs = data.alpha[data.Xa == 1]
    beta_obs  = data.beta[data.Xb == 1]

    plt.hist(alpha_obs, bins=36, label='Trajectoire A (alpha)', color = "#728DB9")
    plt.hist(beta_obs, bins=36, label='Trajectoire B (beta)', color = "#F1A87E")
    plt.legend()
    plt.xlabel('Angles alpha/beta en degrés')
    plt.ylabel('Nombre de clics')
    plt.title('Événements avec détection simple de chaque côté')

def proba_X(data):
    # Angles uniques
    alpha_vals = np.sort(data.alpha.unique())
    beta_vals  = np.sort(data.beta.unique())

    # Probabilités
    p_alpha = [np.mean(data.Xa[data.alpha == a]) for a in alpha_vals]
    p_beta  = [np.mean(data.Xb[data.beta == b]) for b in beta_vals]

    # Largeur des barres
    width = (alpha_vals[1] - alpha_vals[0]) * 0.4

    plt.bar(alpha_vals - width/2, p_alpha, width=width, label='P(Xa=1 | α)', color="#728DB9")
    plt.bar(beta_vals  + width/2, p_beta,  width=width, label='P(Xb=1 | β)', color="#F1A87E")

    plt.legend()
    plt.xlabel('Angles alpha/beta en degrés')
    plt.ylabel('Probabilité de clic')
    plt.title('Probabilité de détection en fonction de l’angle')
    plt.ylim(0,1)