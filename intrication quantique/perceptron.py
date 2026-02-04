# Bibliothèques.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


def perceptron(data, alea):
    # Angles réguliers : extraire les valeurs uniques, compter les détections.
    if not alea:
        angles = np.unique(data.alpha) # Ou directement data.alpha.unique(), moins générique.
        nb_angles = angles.shape[0]
        proba = np.empty((nb_angles,))
        for k, alpha_k in enumerate(angles):
            select_alpha_k = (data.alpha == alpha_k)
            proba[k] = (select_alpha_k & (data.X == 1)).sum() / select_alpha_k.sum()

        plt.scatter(angles, proba, label = 'Angles à intervalles réguliers (empirique)')

    # Angles aléatoires : faire l'histogramme des détections dans des intervalles d'angles.
    # Les angles seront les centres des intervalles.
    else:
        nb_angles = 180
        angles_nb_per_bin, angles_bins = np.histogram(data.alpha, bins = nb_angles, range = (0, 180))
        angles = (angles_bins[1:] + angles_bins[:-1]) * 0.5
        proba = np.empty((nb_angles,))
        for k in range(nb_angles):
            select_alpha_k = ((data.alpha >= angles_bins[k]) & (data.alpha < angles_bins[k+1]))
            proba[k] = (select_alpha_k & (data.X == 1)).sum() / angles_nb_per_bin[k]
        plt.scatter(angles, proba, label = 'Angles aléatoires binnés(empirique)')

    # Partitionner en jeux de données d'entraînement et de test.
    # Le tableau de l'angle doit être 2D (car l'interface doit pouvoir
    # fonctionner avec plusieurs paramètres variant simultanément).
    train_angles, test_angles, train_obs, test_obs = train_test_split(
        np.radians(angles).reshape(-1, 1),
        proba,
        shuffle = True # Vrai par défaut ; essayez avec False pour voir...
    )

    # Jeux de test initialisés ci-dessus, ainsi que plot_angles.
    # Initialisation de la validation croisée.
    grid = GridSearchCV(MLPRegressor(solver = 'adam',
                                    hidden_layer_sizes=(30, 30),
                                    max_iter = 2000),
                        {'alpha': 10. ** -np.arange(1, 5)},
                        cv = 3)
    grid.fit(train_angles, train_obs)
    print(grid.best_estimator_)

    # Tester plusieurs fois, tracer la loi de probabilité prédite.
    plot_angles = np.linspace(0, 180)
    for k in range(1, 9):
        grid.fit(train_angles, train_obs)
        print(f'Score {k} = {grid.score(test_angles, test_obs)}')
        plt.plot(plot_angles, grid.predict(np.radians(plot_angles).reshape(-1, 1)),
                label = f'Itération {k}')

    _ = plt.legend()



def main():
    # Préchargement des jeux de données.
    df_1photon_x = pd.read_csv('jour1-1-facile/1photon-polar-x-alea.csv', sep=';')
    #df_1photon_x_angles_alea = pd.read_csv('datasets/jour1-1-facile/1photon-polar-x-alea.csv', sep=';')
    #df_1photon_y = pd.read_csv('datasets/jour1-1-facile/1photon-polar-y-alea.csv', sep=';')
    df_1photon_30 = pd.read_csv('jour1-1-facile/1photon-polar-30-intervalles.csv', sep=';')

    data=df_1photon_30
    alea=False
    perceptron(data, alea)
    plt.show()


main()