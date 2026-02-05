import math
import numpy as np
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split

# Fonction de comptage des mesures et des coïncidences (XX, XY, ...)
# ainsi que la quantité E, dans des tableaux 2D indexés par les angles
# (alpha sur l'axe 0, beta sur l'axe 1).
def count_detections(data, angles):
    nb_angles = len(angles)
    (count_measurements, count_xx, count_xy, count_yx, count_yy) = (
        np.zeros((nb_angles, nb_angles)) for _ in range(5)
    )
    for i, alpha in enumerate(angles):
        for j, beta in enumerate(angles):
            select_angles = (data.alpha == alpha) & (data.beta == beta)
            count_measurements[i, j] = select_angles.sum()
            (Xa, Xb, Ya, Yb) = (data[name][select_angles] == 1 for name in ('Xa', 'Xb', 'Ya', 'Yb'))
            count_xx[i, j] = (Xa & Xb).sum()
            count_xy[i, j] = (Xa & Yb).sum()
            count_yx[i, j] = (Ya & Xb).sum()
            count_yy[i, j] = (Ya & Yb).sum()

    E = (count_xx + count_yy - count_yx - count_xy) / count_measurements
    return (count_measurements, count_xx, count_xy, count_yx, count_yy, E)

# Fonction de calcul de S à partir du tableau E.
def S_from_E(E):
    n = E.shape[0]
    assert E.shape == (n, n), f'Le tableau E doit être carré, pas de forme {E.shape}'
    S = (E.reshape(n, 1, n, 1)   # alpha, beta
         + E.reshape(1, n, n, 1) # alpha', beta
         - E.reshape(n, 1, 1, n) # alpha, beta'
         + E.reshape(1, n, 1, n) # alpha', beta'
        )
    return S

# Pour le jeu de données testé, recherche des angles où |S| est max
# sur une partie des données.
def eval_intric(data):
    angles = np.unique(data[['alpha', 'beta']])
    nb_angles = len(angles)
    data_search_S, data_test_S = train_test_split(data, train_size = 0.25)
    E = count_detections(data_search_S, angles)[-1]
    absS = np.abs(S_from_E(E))
    S_max = absS[np.isfinite(absS)].max() # Éviter les valeurs 0/0 avec isfinite().
    (i_alpha1, i_alpha2, i_beta1, i_beta2) = np.argwhere(absS == S_max)[0]
    (alpha1, alpha2, beta1, beta2) = (angles[i] for i in (i_alpha1, i_alpha2, i_beta1, i_beta2))
    #print(alpha1, alpha2, beta1, beta2)

    # Calcul d'une borne de la p-valeur de |S| sur le reste des données
    # pour les angles trouvés où |S| était max ci-dessus.  Pour le nombre
    # de mesures, au cas où ce ne serait pas le même pour les différentes
    # combinaisons d'angles, vu qu'on calcule une borne, on peut prendre
    # le min.
    (counts, _, _, _, _, E) = count_detections(data_test_S, angles)
    S = E[i_alpha1, i_beta1] + E[i_alpha2, i_beta1] - E[i_alpha1, i_beta2] + E[i_alpha2, i_beta2]
    num = min(counts[i_alpha1, i_beta1], counts[i_alpha2, i_beta1],
            counts[i_alpha1, i_beta2], counts[i_alpha2, i_beta2])
    pval = 2 * math.exp(-((abs(S) - 2)**2 * num) / 16)
    print(f'|S| max = {abs(S)}, pvalue <= {pval}')
    return abs(S) > 2.1