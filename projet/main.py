### Code principal qui a pour fonction de faire dans l'odre:
# Afficher les histogrammes donnant les probabilités de détection
# Tester si les photons sont intriqués
# S'ils sont indépendants, vérifier s'ils suivent des lois en cos2(alpha-theta)
# S'ils suivent une telle loi, la vérifier en faisant une régression linéaire

# Bibliothèqes standards
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#Import de foncions utiles
import hist
import eval_intric
import eval_model

def test_unitaire(data): #version graphique et jolie de l'analyse des fichiers
    ######## Vérification de la conformité du jeu de données et mise en forme
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
    hist.proba_X(data)

    intric=eval_intric.eval_intric(data) # True si les photons sont intriqués

    if intric:
        print("Les photons sont intriqués")
    else:
        print("Les photons sont indépendants")
        (theta1, theta2) = eval_model.theta_ideal(data)
    
    plt.show()

def analyse(data):
    pvalue_correlation, S = eval_intric.eval_intric(data)
    intrication = S > 2.1
    ((angle1, _, _ , _), (angle2, _, _, _)) = eval_model.theta_model(data) #Méthode de Fischer ne marche pas pour df_xx
    pass
    #return angle1,pvalue1,angle2,pvalue2,pvalue_correlation,intrication


def projet(nom_dossier):
    dossier = Path(nom_dossier)
    nb_fichiers = sum(1 for f in dossier.iterdir() if f.is_file())
    print(f"fichiers allant de 00 à {nb_fichiers-1}")

    réponses=[]
    for i in range(nb_fichiers):
        if i < 10:
            data = pd.read_csv(f"./test-projet/{nom_dossier}/dataset-0{i}.csv", sep=';')
        else:
            data = pd.read_csv(f"./test-projet/{nom_dossier}/dataset-{i}.csv", sep=';')
        réponses.append(analyse(data))


if __name__ == "__main__":
    ######## Chargement des jeux de données
    
    #jour2
    # Batteries de test sans bruit
    #df_45_1M=pd.read_csv("./jour2-1-facile/2photons-45-1M.csv", sep=';')
    #df_autre_1M=pd.read_csv("./jour2-1-facile/2photons-autre-1M.csv", sep=';')
    #df_circ_1M=pd.read_csv("./jour2-1-facile/2photons-circ-1M.csv", sep=';')
    #df_epr_1M=pd.read_csv("./jour2-1-facile/2photons-epr-1M.csv", sep=';')
    #df_xx_1M=pd.read_csv("./jour2-1-facile/2photons-xx-1M.csv", sep=';')
    #df_yy_1M=pd.read_csv("./jour2-1-facile/2photons-yy-1M.csv", sep=';')
    
    #  Batteire de tests avec bruit
    df_epr=pd.read_csv("./jour2-2-moyen/2photons-epr.csv", sep=';')
    df_xx=pd.read_csv("./jour2-2-moyen/2photons-xx.csv", sep=';')
    df_yy=pd.read_csv("./jour2-2-moyen/2photons-yy.csv", sep=';')

    #Test unitaire
    data = df_xx
    test_unitaire(data)

    # Test projet
    nom_dossier = "polar-2photons-1sozuv"
    projet(nom_dossier)
