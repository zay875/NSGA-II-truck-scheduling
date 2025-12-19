from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

'''# --- Données de base ---
H, D, K, N = 5, 2, 2, 4
Y = 6
Cd = [705, 608]
CE = 0.5
'''

# Charger les CSV (chemins relatifs)
containers_df = pd.read_csv("example/containers_all.csv")
trucks_df = pd.read_csv("example/trucks_all.csv")
docks_df = pd.read_csv("example/docks_all.csv")
param_df = pd.read_csv("example/parameters_all.csv")
# Récupérer toutes les instances existantes
instances = sorted(containers_df["Instance"].unique())
 
# --- Fonction distance (absolue ici, car positions sont scalaires) ---
def distance(p1, p2):
    return abs(p1 - p2)

for instance_id in instances:
        # Filtrage des données
    cont_i = containers_df[containers_df["Instance"] == instance_id].copy()
    trucks_i = trucks_df[trucks_df["Instance"] == instance_id].copy()
    docks_i = docks_df[docks_df["Instance"] == instance_id].copy()

    params_i=param_df[param_df["Instance"]== instance_id].copy()

        # Extraire automatiquement les paramètres présents dans le CSV
    params = dict(zip(params_i["Parameter"], params_i["Value"]))

    # Liste des paramètres que tu veux garder
    keep_params = ["H", "D", "N", "K", "Y", "CE", "I", "V"]

    # Créer un dictionnaire filtré
    params_filtered = {p: params[p] for p in keep_params if p in params}

    # Conversion automatique des valeurs (float → int quand c’est un entier)
    for key, val in params_filtered.items():
        if float(val).is_integer():
            params_filtered[key] = int(val)
        else:
            params_filtered[key] = float(val)

    # Affectation directe
    H = params_filtered["H"]
    D = params_filtered["D"]
    N = params_filtered["N"]
    K = params_filtered["K"]
    Y = params_filtered["Y"]
    CE = params_filtered["CE"]
    I = params_filtered["I"]
    V = params_filtered["V"]

    # --- F1 : fonction de coût ---
    def F1(x):
        """
        Objectif 1 : coût total de transport basé sur l’utilisation des camions et leurs destinations.
        """
        # Variables binaires : a, xhk, pih
        a = [1 if v > 0.5 else 0 for v in x[0:H*D]]
        xhk = [1 if v > 0.5 else 0 for v in x[H*D:H*D + H*K]]
        vh  = [1 if v > 0.5 else 0 for v in x[H*D + H*K:H*D + H*K + H]]
        pih = [1 if v > 0.5 else 0 for v in x[H*D + H*K + H:H*D + H*K + H + N*H]]
        pih = np.array(pih).reshape(N, H)

        # Coût lié aux destinations (Cd)
        total_cost = 0


        penalty = 0
        for h in range(H):
            truck = trucks_i.iloc[h]
        
            # Pénalité si deux camions ont le même dock
            if trucks_i["DockPosition"].duplicated().any():
                penalty += 1000

            # Coût de destination
            for d in range(D):
                total_cost += truck["Cost"] * a[h*D + d]

            # Pénalité si ce camion n’est pas utilisé
            if not np.any(pih[:, h] == 1):
                penalty += 2000

            # si vh[h]==1 mais aucune destination active → pénalité
            if vh[h] == 1 and sum(a[h*D + d] for d in range(D)) == 0:
                penalty += 1000
            # si vh[h]==0 mais une destination est active → pénalité
            if vh[h] == 0 and sum(a[h*D + d] for d in range(D)) > 0:
                penalty += 1000

        cost_F1 = total_cost + penalty
        return cost_F1


    # --- F2 : fonction d’énergie / distance ---
    def F2(x):
        """
        Objectif 2 : minimiser la distance totale (énergie) 
        + pénalités si contraintes non respectées.
        """
        a = [1 if v > 0.5 else 0 for v in x[0:H*D]]
        xhk = [1 if v > 0.5 else 0 for v in x[H*D:H*D + H*K]]
        vh  = [1 if v > 0.5 else 0 for v in x[H*D + H*K:H*D + H*K + H]]
        pih = [1 if v > 0.5 else 0 for v in x[H*D + H*K + H:H*D + H*K + H + N*H]]
        pih = np.array(pih).reshape(N, H)

        total_energy = 0
        penalty = 0

        for i in range(N):  # chaque conteneur
            container = cont_i.iloc[i]
            assigned_trucks = np.where(pih[i, :] == 1)[0]

            # (1) Chaque conteneur doit être affecté à un seul camion
            if len(assigned_trucks) != 1:
                penalty += 1000  # grosse pénalité
            # verify that each container is assigned to one dock pposition
            for h in assigned_trucks:
                truck = trucks_i.iloc[h]

                # (2) Vérifier la correspondance des destinations
                if container["Destination"] != truck["Destination"]:
                    penalty += 1000
                #chaque camion ne dessert que les conteneurs avce la meme destination
                if pih[i, h] == 1:
                    d = int(cont_i.iloc[i]["Destination"]) - 1
                    if a[h*D + d] == 0:
                        penalty += 2000
                # Énergie = 2*distance + Y*longueur
                Pi = container["Position"]
                Rk = truck["DockPosition"]
                Li = container["Length"]
                total_energy += 2 * abs(Pi - Rk) + Y * Li

        # (3) Capacité maximale de chaque camion
        for h in range(H):
            truck = trucks_i.iloc[h]
            load = sum(cont_i.iloc[i]["Length"] for i in range(N) if pih[i, h] == 1)
            if load > truck["Capacity"]:
                penalty += (load - truck["Capacity"]) * 1000  # pénalité proportionnelle

        # (4) Total F2 = énergie + pénalités
        F2_value = CE * total_energy + penalty
        return F2_value

    # --- Définition du problème NSGA-II ---
    num_vars = H*D + H*K + H + N*H + N*H

    # --- définir les plages par bloc ---
    range_a = [(0, 1)] * (H * D)       # 10
    range_xhk = [(0, 1)] * (H * K)     # 10
    range_vh = [(0, 1)] * (H)          # 5
    range_pih = [(0, 1)] * (N * H)     # 20
    range_z = [(0, 100)] * (N * H)     # 20

    var_ranges = range_a + range_xhk + range_vh + range_pih + range_z



    problem = Problem(
        num_of_variables=num_vars,
        objectives=[F1, F2],
        variables_range=var_ranges
    )

    # --- Exécution de l’évolution ---
    evo = Evolution(problem)
    final_pop = evo.evolve()

    # --- Extraction des résultats ---
    data = [{
        **{f"x{i}": val for i, val in enumerate(ind.features)},
        "F1": ind.objectives[0],
        "F2": ind.objectives[1]
    } for ind in final_pop]

    df = pd.DataFrame(data)
    print(df.head())

    # --- Sauvegarde optionnelle ---
    df.to_csv(f"results_population_instances_{instance_id}.csv", index=False)

    # --- Visualisation du front de Pareto ---
    # --- Visualisation et sauvegarde du front de Pareto pour cette instance ---
    funcs = [ind.objectives for ind in final_pop]
    f1_vals = [f[0] for f in funcs]
    f2_vals = [f[1] for f in funcs]

    plt.figure()
    plt.scatter(f1_vals, f2_vals, color='blue', alpha=0.7)
    plt.xlabel("F1 (Coût total)")
    plt.ylabel("F2 (Énergie totale)")
    plt.title(f"Front de Pareto - Instance {instance_id}")

    # Enregistrer l’image dans un dossier de résultats
    output_dir = "pareto_plots"
    os.makedirs(output_dir, exist_ok=True)  # crée le dossier s'il n'existe pas

    filename = os.path.join(output_dir, f"pareto_instance_{instance_id}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # ferme la figure pour libérer la mémoire

    print(f"✅ Graphique Pareto sauvegardé : {filename}")


