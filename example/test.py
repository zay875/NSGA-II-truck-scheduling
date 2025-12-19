from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Données de base ---
H, D, K, N = 5, 2, 2, 4
Y = 6
Cd = [705, 608]
CE = 0.5

# --- Données containers et camions ---
containers_data = {
    "id": ["C1", "C2", "C3", "C4"],
    "length": [5, 4, 5, 4],  # taille en mètres
    "destination": ["2", "1", "2", "2"],
    "position": [25, 7, 29, 28]  # position sur le quai
} 
containers_df = pd.DataFrame(containers_data)

trucks_data = {
    "id": ["T1", "T2", "T3","T4","T5"],
    "capacity": [6, 6, 6,6, 6],
    "destination": ["2", "1", "2","1","1"],
    "dock_position": [2, 4, 2,5,6],
    "cost_destination": [608,705,608,705,705]
}
trucks_df = pd.DataFrame(trucks_data)


 
# --- Fonction distance (absolue ici, car positions sont scalaires) ---
def distance(p1, p2):
    return abs(p1 - p2)


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
        truck = trucks_df.iloc[h]
    
        # Pénalité si deux camions ont le même dock
        if trucks_df["dock_position"].duplicated().any():
         penalty += 1000

        # Coût de destination
        for d in range(D):
            total_cost += truck["cost_destination"] * a[h*D + d]

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
        container = containers_df.iloc[i]
        assigned_trucks = np.where(pih[i, :] == 1)[0]

        # (1) Chaque conteneur doit être affecté à un seul camion
        if len(assigned_trucks) != 1:
            penalty += 1000  # grosse pénalité
        # verify that each container is assigned to one dock pposition
        for h in assigned_trucks:
            truck = trucks_df.iloc[h]

            # (2) Vérifier la correspondance des destinations
            if container["destination"] != truck["destination"]:
                penalty += 1000
            #chaque camion de dessert que les conteneurs avce la meme destination
            if pih[i, h] == 1:
                d = int(containers_df.iloc[i]["destination"]) - 1
                if a[h*D + d] == 0:
                    penalty += 2000
            # Énergie = 2*distance + Y*longueur
            Pi = container["position"]
            Rk = truck["dock_position"]
            Li = container["length"]
            total_energy += 2 * abs(Pi - Rk) + Y * Li

    # (3) Capacité maximale de chaque camion
    for h in range(H):
        truck = trucks_df.iloc[h]
        load = sum(containers_df.iloc[i]["length"] for i in range(N) if pih[i, h] == 1)
        if load > truck["capacity"]:
            penalty += (load - truck["capacity"]) * 1000  # pénalité proportionnelle

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
df.to_csv("results_population_truck_det_constraint.csv", index=False)

# --- Visualisation du front de Pareto ---
funcs = [ind.objectives for ind in final_pop]
f1_vals = [f[0] for f in funcs]
f2_vals = [f[1] for f in funcs]

plt.figure()
plt.xlabel("F1 (cost)")
plt.ylabel("F2 (energy)")
plt.scatter(f1_vals, f2_vals)
plt.title("Pareto Front (NSGA-II)")
plt.show()
