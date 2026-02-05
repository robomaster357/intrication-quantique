import numpy as np
import pandas as pd # type: ignore
import scipy.stats as sps
from scipy.stats import chi2

# Définir une fonction pour la loi de probabilité, en fonction des angles en degrés.
# La fonction peut opérer sur des nombres ou des tableaux.
def f_proba(alpha, theta):
    return np.cos(np.radians(alpha - theta)) ** 2

# Fonction pour calculer la valeur-p pour un theta spécifique.
def pvalue(theta):
    global table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    # Probabilité attendue pour chaque angle.
    p0 = f_proba(table_alpha_uniques, theta)

    # Valeur-p pour chaque angle.
    pvalues = [2*(1 - sps.norm.cdf(abs(p - x)*(n/(p*(1-p)))**0.5))
               for p,x,n in zip(moyenne_obs_alpha, p0, nb_alpha)]

    # Comme p-values E [0 ; 1], on commence par borner pavalues entre eps (proche de 0) et 1 pour éviter les valeurs inf dans le log
    eps = np.finfo(float).tiny  # ≈ 2e-308
    pvalues = np.clip(pvalues, eps, 1)

    # Combinaison des valeurs-p par la méthode de Fisher.
    Tchi = -2 * np.sum(np.log(pvalues))

    # Retrouver la valeur-p correspondante via la distribution chi²
    # avec 2*(nombre de valeurs-p) degrés de liberté.
    return 1 - sps.chi2.cdf(Tchi, 2*len(table_alpha_uniques))

def eval_theta(data):
    global table_alpha, obs, test_theta, table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    table_alpha = data.alpha
    obs = data.X
    test_theta = np.arange(0, 180, 0.1)

    # Calcul des valeurs-p pour chaque angle de mesure rencontré,
    # à partir du nombre de fois où on l'a rencontré et de la valeur
    # moyenne de l'observation en X.
    table_alpha_uniques = np.unique(table_alpha)
    nb_alpha = np.asarray([len(table_alpha[table_alpha == a]) for a in table_alpha_uniques])
    moyenne_obs_alpha = np.asarray([np.mean(obs[table_alpha == a]) for a in table_alpha_uniques])
    #t = sps.norm.ppf(0.05) # FIXME this was in the old lab, not used?

    # Le bloc suivant élimine les angles pour lesquels les probabilités
    # sont trop hautes ou trop basses.  Essayez sans pour voir ;
    # pourquoi posent-elles problème ?
    indices_alpha_ok = (moyenne_obs_alpha > 0.1) & (moyenne_obs_alpha < 0.9)
    table_alpha_uniques = table_alpha_uniques[indices_alpha_ok]
    moyenne_obs_alpha = moyenne_obs_alpha[indices_alpha_ok]
    nb_alpha = nb_alpha[indices_alpha_ok]

    # Intervalle de confiance.
    table_pvalues = np.asarray([pvalue(theta) for theta in test_theta])
    #print(table_pvalues[table_pvalues > 0.05])
    indices_pvaleur_grande = np.where(table_pvalues > 0.05)[0]
    CI = [float(test_theta[indices_pvaleur_grande[0]]), float(test_theta[indices_pvaleur_grande[-1]])]
    return CI

def theta(data):
    global table_alpha, obs, test_theta, table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    data1 = data[['alpha', 'Xa', 'Ya']]
    data1 = data1.rename(columns={'alpha': 'alpha', 'Xa':'X', 'Ya':'Y'})
    data2 = data[['beta', 'Xb', 'Yb']]
    data2 = data2.rename(columns={'beta': 'alpha', 'Xb':'X', 'Yb':'Y'})
    theta1 = eval_theta(data1) #theta_alpha appartenant à l'intervalle de confiance si existence
    print(f"Estimation de theta pour alpha entre : [{theta1[0]} ; {theta1[1]}]")
    theta2 = eval_theta(data2) #theta_beta appartenant à l'intervalle de confiance si existence
    print(f"Estimation de theta pour beta entre : [{theta2[0]} ; {theta2[1]}]")
    return (theta1, theta2)