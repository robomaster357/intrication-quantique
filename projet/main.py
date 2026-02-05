### Code principal qui a pour fonction de faire dans l'odre:
# Afficher les histogrammes donnant les probabilités de détection
# Tester si les photons sont intriqués
# S'ils sont indépendants, vérifier s'ils suivent des lois en cos2(alpha-theta)
# S'ils suivent une telle loi, la vérifier en faisant une régression linéaire

# Bibliothèqes standards
import pandas as pd
import matplotlib.pyplot as plt

#Import de foncions utiles
import hist
import eval_intric
import eval_model

if __name__ == "__main__":
    #Chargement des jeux de données
    #jour2-1
    df_45_1M=pd.read_csv("./jour2-1-facile/2photons-45-1M.csv", sep=';')
    df_autre_1M=pd.read_csv("./jour2-1-facile/2photons-autre-1M.csv", sep=';')
    df_circ_1M=pd.read_csv("./jour2-1-facile/2photons-circ-1M.csv", sep=';')
    df_epr_1M=pd.read_csv("./jour2-1-facile/2photons-epr-1M.csv", sep=';')
    df_xx_1M=pd.read_csv("./jour2-1-facile/2photons-xx-1M.csv", sep=';')
    df_yy_1M=pd.read_csv("./jour2-1-facile/2photons-yy-1M.csv", sep=';')
    df_epr=pd.read_csv("./jour2-2-moyen/2photons-epr.csv", sep=';')
    df_xx=pd.read_csv("./jour2-2-moyen/2photons-xx.csv", sep=';')
    df_yy=pd.read_csv("./jour2-2-moyen/2photons-yy.csv", sep=';')


    #Choix du jeu de données
    data=df_circ_1M

    #### Vérification de la conformité du jeu de données et mise en forme
    (_, nb_col) = data.shape
    assert(nb_col == 6), "Vérifier le nombre de colonnes du jeu de donées" # Bon nombre de colonnes
    
    col=list(data.columns)
    data = data.rename(columns={col[0]: 'alpha', col[1]: 'beta', col[2]: 'Xa', col[3]:'Ya', col[4]:'Xb', col[5]:'Yb'}) # Renommage des colonnes pour faciliter le traitement de données
    
    valid = (
        ((data.Xa + data.Ya) == 1) &  # côté A : un seul détecteur
        ((data.Xb + data.Yb) == 1)    # côté B : un seul détecteur
    )
    data = data[valid] #sélection des données conformes au domaine étudié

    # Affichage des histogrammes
    hist.aff_hist(data)
    plt.show()

    intric=eval_intric.eval_intric(data) # True si les photons sont intriqués

    if intric:
        print("Les photons sont intriqués")
    else:
        print("Les photons sont indépendants")
        (theta1, theta2) = eval_model.theta(data)
    
