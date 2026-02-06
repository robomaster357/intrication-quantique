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

    S, pval=eval_intric.eval_intric(data) # True si les photons sont intriqués
    print(S, pval)
    if S > 2.1:
        print("Les photons sont intriqués")
    else:
        print("Les photons sont indépendants")
    #((theta1, A1, B1, rmse1, pavalue1), (theta2, A2, B2, rmse2, pvalue2)) = eval_model.theta_model(data)
    #print(theta1, rmse1)
    #print(theta2, rmse2)
    (theta1, ptheta1, theta2, ptheta2) = eval_model.theta_ideal(data)
    print(theta1, ptheta1)
    print(theta2, ptheta2)
    plt.show()

def analyse(data):
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

    ######## Extraction des différents paramètres
    S, pvalue_correlation = eval_intric.eval_intric(data)
    viole_CHSH = S > 2.78
    intrication = S > 2.2
    (angle1, pvalue1, angle2, pvalue2) = eval_model.theta_ideal(data) #Méthode de Fischer ne marche pas pour df_xx
    return angle1,pvalue1,angle2,pvalue2,pvalue_correlation,S,intrication, viole_CHSH


def projet(nom_dossier):
    dossier = Path(f"./test-projet/{nom_dossier}")
    nb_fichiers = sum(1 for f in dossier.iterdir() if f.is_file())
    print(f"fichiers allant de 00 à {nb_fichiers-1}")

    #Création du fichier csv
    colonnes=["fichier", "angle1", "pvalue1", "angle2", "pvalue2", "pvalue_correlation", "S", "intrication", "viole CHSH"]
    ans = pd.DataFrame(columns=colonnes)
    ans.to_csv(f"polar-2photons-{nom_dossier}.csv", index=False)

    for i in range(nb_fichiers):
        if i < 10:
            data = pd.read_csv(f"./test-projet/{nom_dossier}/dataset-0{i}.csv", sep=';')
            angle1,pvalue1,angle2,pvalue2,pvalue_correlation,S,intrication, CHSH = analyse(data)
            nouvelle_ligne = {"fichier": f"dataset-0{i}.csv", "angle1": angle1, "pvalue1": pvalue1, "angle2": angle2, "pvalue2": pvalue2, "pvalue_correlation": pvalue_correlation, "S": S, "intrication": intrication, "viole CHSH": CHSH}
        else:
            data = pd.read_csv(f"./test-projet/{nom_dossier}/dataset-{i}.csv", sep=';')
            angle1,pvalue1,angle2,pvalue2,pvalue_correlation,intrication, CHSH = analyse(data)
            nouvelle_ligne = {"fichier": f"dataset-{i}.csv", "angle1": angle1, "pvalue1": pvalue1, "angle2": angle2, "pvalue2": pvalue2, "pvalue_correlation": pvalue_correlation, "S": S, "intrication": intrication, "viole CHSH": CHSH}
        ans = pd.read_csv(f"polar-2photons-{nom_dossier}.csv")
        ans.loc[len(ans)] = nouvelle_ligne
        ans.to_csv(f"polar-2photons-{nom_dossier}.csv", index=False)
        print(f"fichier {i} analysé, S = {S}")

def quiz(nom_dossier):
    alpha = 0.05

    projet(nom_dossier)

    ans = pd.read_csv(f"polar-2photons-{nom_dossier}.csv")

    # Conditions polarisation identifiable
    #polA = ans["pvalue1"] > alpha
    #polB = ans["pvalue2"] > alpha

    polA = ans["pvalue1"] != -1
    polB = ans["pvalue2"] != -1

    # Polarisations identiques
    rep1 = ((polA) & (polB) & (abs(ans["angle1"] - ans["angle2"]) < 2)).sum()

    # Polarisations différentes
    rep2 = ((polA) & (polB) & (abs(ans["angle1"] - ans["angle2"]) > 2)).sum()

    # Corrélation identifiable
    #rep3 = (ans["pvalue_correlation"] < alpha).sum()
    rep3 = (ans["intrication"] == True).sum()
    # Violation CHSH
    rep4 = (ans["viole CHSH"] == True).sum()

    print("Nombre de fichiers avec :")
    print("avec 2 polarisations linéaires identifiables (p-valeur > 0.05) identiques :", rep1)
    print("avec 2 polarisations linéaires identifiables (p-valeur idem) différentes :", rep2)
    print("présentant une corrélation identifiable (p-valeur idem) sur les coïncidences entre les photons A et B :", rep3)
    print("présentant une violation de l'inégalité CHSH :", rep4)


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
    #df_epr=pd.read_csv("./jour2-2-moyen/2photons-epr.csv", sep=';')
    #df_xx=pd.read_csv("./jour2-2-moyen/2photons-xx.csv", sep=';')
    #df_yy=pd.read_csv("./jour2-2-moyen/2photons-yy.csv", sep=';')

    #Test unitaire
    #data = pd.read_csv("./test-projet/polar-2photons-sbc4jm/dataset-75.csv", sep=';')
    #test_unitaire(data)

    # Test projet
    nom_dossier = "polar-2photons-dc660a"
    quiz(nom_dossier)
