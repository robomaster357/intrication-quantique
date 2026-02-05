import matplotlib.pyplot as plt
import pandas as pd

def aff_hist(data):
    alpha_obs = data.alpha[data.Xa == 1]
    beta_obs  = data.beta[data.Xb == 1]

    plt.hist(alpha_obs, bins=36, label='Trajectoire A (alpha)', color = "#728DB9")
    plt.hist(beta_obs, bins=36, label='Trajectoire B (beta)', color = "#F1A87E")
    plt.legend()
    plt.xlabel('Angles alpha/beta en degrés')
    plt.ylabel('Nombre de clics')
    plt.title('Événements avec détection simple de chaque côté')