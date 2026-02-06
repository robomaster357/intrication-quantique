import numpy as np
import pandas as pd # type: ignore
import scipy.stats as sps
from scipy.stats import chi2
from scipy.optimize import curve_fit
from scipy import stats


###################################### Evaluation de théta par méthode de Fischer ################################

# Définir une fonction pour la loi de probabilité, en fonction des angles en degrés.
# La fonction peut opérer sur des nombres ou des tableaux.
def f_proba_ideal(alpha, theta):
    return np.cos(np.radians(alpha-theta))**2

# Fonction pour calculer la valeur-p pour un theta spécifique.
def pvalue(theta):
    global table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    # Probabilité attendue pour chaque angle.
    p0 = f_proba_ideal(table_alpha_uniques, theta)

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

def eval_theta_ideal(data):
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
    max_pvalues = np.max(table_pvalues)
    indice_max_pavalues = np.argmax(table_pvalues)
    if indice_max_pavalues > 0.05 :
        return test_theta[indice_max_pavalues], max_pvalues
    else:
        return -1, -1
    """
    indices_pvaleur_grande = np.where(table_pvalues > 0.05)[0]
    if len(indices_pvaleur_grande)>0: # existence d'un intervalle de theta vérifiant p-valeur > 0.05
        
        print(test_theta[indices_pvaleur_grande[-1]], test_theta[indices_pvaleur_grande[0]])
        if abs(test_theta[indices_pvaleur_grande[-1]]-test_theta[indices_pvaleur_grande[0]]) < 5 : 
            assert(abs(test_theta[indices_pvaleur_grande[-1]]-test_theta[indices_pvaleur_grande[0]]) < 5), "Evaluation de theta trop imprécise"
            CI = [float(test_theta[indices_pvaleur_grande[0]]), float(test_theta[indices_pvaleur_grande[-1]])]
        else: 
            CI = [[0, test_theta[indices_pvaleur_grande[0]]], [test_theta[indices_pvaleur_grande[-1]], 180]] # gère le cas ou theta oscille entre 0 et 180 degrés
        return CI
    
        return test_theta[indices_pvaleur_grande[0]], np.max(table_pvalues)
    else:
        return (-1, -1)
    """

def theta_ideal(data):
    global table_alpha, obs, test_theta, table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    data1 = data[['alpha', 'Xa', 'Ya']]
    data1 = data1.rename(columns={'alpha': 'alpha', 'Xa':'X', 'Ya':'Y'})
    data2 = data[['beta', 'Xb', 'Yb']]
    data2 = data2.rename(columns={'beta': 'alpha', 'Xb':'X', 'Yb':'Y'})
    theta1, ptheta1 = eval_theta_ideal(data1) #theta_alpha appartenant à l'intervalle de confiance
    
    """if theta1:
        if isinstance(theta1[0], float):
            print(f"Estimation de theta pour alpha entre : [{theta1[0]} ; {theta1[1]}]")
        else:
            print(f"Estimation de theta pour alpha entre : d'une part [{theta1[0][0]} ; {theta1[0][1]}] et d'autre part [{theta1[1][0]} ; {theta1[1][1]}]")
    else:
        print("Aucun θ compatible avec une loi cos² — modèle rejeté pour alpha.")"""
    
    theta2, ptheta2 = eval_theta_ideal(data2) #theta_beta appartenant à l'intervalle de confiance
    """if theta2:
        if isinstance(theta2[0], float):
            print(f"Estimation de theta pour beta entre : [{theta2[0]} ; {theta2[0]}]")
        else:
            print(f"Estimation de theta pour beta entre : d'une part [{theta2[0][0]} ; {theta2[0][1]}] et d'autre part [{theta2[1][0]} ; {theta2[1][1]}]")
    else:
        print("Aucun θ compatible avec une loi cos² — modèle rejeté pour beta.")"""
    return (theta1, ptheta1, theta2, ptheta2)









###################################### Evaluation de théta par méthode des moindres carrés ################################

def f_proba_model(alpha, theta, A, B):
    # Modèle réaliste : visibilité A + bruit B
    return A * np.cos(np.radians(alpha - theta))**2 + B

"""
def eval_theta_model(data):
    global table_alpha, obs, test_theta, table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    table_alpha = data.alpha
    obs = data.X

    # Moyennes observées pour chaque angle
    table_alpha_uniques = np.unique(table_alpha)
    moyenne_obs_alpha = np.array([np.mean(obs[table_alpha == a]) for a in table_alpha_uniques])

    # Ajustement par moindres carrés
    # Bornes physiques : 0 ≤ A ≤ 1, 0 ≤ B ≤ 0.5, 0 ≤ θ ≤ 180
    params, _ = curve_fit(
        f_proba_model,
        table_alpha_uniques,
        moyenne_obs_alpha,
        bounds=([0, 0, 0], [180, 1, 0.5])
    )

    theta_est, A_est, B_est = params

    #Evaluation des résidus pour vérifier la validité du modèle
    residus = moyenne_obs_alpha - f_proba_model(table_alpha_uniques, *params)
    rmse = np.sqrt(np.mean(residus**2))
    
    return round(theta_est, 0), round(A_est, 2), round(B_est, 2), round(rmse, 3)
"""

def eval_theta_model(data):
    table_alpha = data.alpha.values
    obs = data.X.values

    alpha_vals = np.unique(table_alpha)
    p_obs = np.array([np.mean(obs[table_alpha == a]) for a in alpha_vals])

    # Ajustement
    params, cov = curve_fit(
        f_proba_model,
        alpha_vals,
        p_obs,
        bounds=([0, 0, 0], [180, 1, 0.5])
    )

    theta_est, A_est, B_est = params

    # --- Statistiques ---
    residus = p_obs - f_proba_model(alpha_vals, *params)
    dof = len(p_obs) - len(params)          # degrés de liberté
    mse = np.sum(residus**2) / dof         # variance résiduelle

    se = np.sqrt(np.diag(cov) * mse)       # erreurs standards

    t_theta = theta_est / se[0]            # statistique t
    p_theta = 2 * (1 - stats.t.cdf(abs(t_theta), dof))

    rmse = np.sqrt(np.mean(residus**2))

    #print(f"θ = {theta_est:.2f}° ± {se[0]:.2f}°  (p = {p_theta:.3e})")

    return theta_est, A_est, B_est, rmse, p_theta


def theta_model(data):
    global table_alpha, obs, test_theta, table_alpha_uniques, nb_alpha, moyenne_obs_alpha
    data1 = data[['alpha', 'Xa', 'Ya']]
    data1 = data1.rename(columns={'alpha': 'alpha', 'Xa':'X', 'Ya':'Y'})
    data2 = data[['beta', 'Xb', 'Yb']]
    data2 = data2.rename(columns={'beta': 'alpha', 'Xb':'X', 'Yb':'Y'})

    (theta1, A1, B1, rmse1, pavalue1) = eval_theta_model(data1) #theta_alpha appartenant à l'intervalle de confiance
    """if rmse1 < 0.1: 
        print(f"Pour alpha θ vaut {theta1} -> modèle de la forme {A1}cos²(α-{theta1})+{B1} avec rmse={rmse1} et ptheta={pavalue1}")
    else:
        print(f"Le modèle en cos²(α-θ) est faux: rmse = {rmse1}")"""

    (theta2, A2, B2, rmse2, pvalue2) = eval_theta_model(data2) #theta_alpha appartenant à l'intervalle de confiance
    """if rmse2 < 0.1: 
        print(f"Pour beta θ vaut {theta2} -> modèle de la forme {A2}cos²(β-{theta2})+{B2} avec rmse={rmse2} et ptheta={pvalue2}")
    else:
        print(f"Le modèle en cos²(β-θ) est faux: rmse = {rmse2}")"""
    return ((theta1, A1, B1, rmse1, pavalue1), (theta2, A2, B2, rmse2, pvalue2))