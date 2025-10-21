from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
H, D, K, N = 3, 2, 2, 4
Cd = [705, 608]
CE = 0.5

def F1(x):
    a = x[0:H*D]
    a = [round(v) for v in x[0:H*D]]
    total = 0
    for h in range(H):
        for d in range(D):
            total += Cd[d] * a[h*D + d]
    return total

def F2(x):
    start_z = H*D + H*K + N*H  # adjust if you include x,p variables
    z = x[start_z:start_z + (N*H)]
    return CE * sum(z)

num_vars = H*D + H*K + N*H + N*H  # depending on which variables you include
var_ranges = [(0, 10)] * num_vars  # for binary; you can clip later

problem = Problem(num_of_variables=num_vars,
                  objectives=[F1, F2],
                  variables_range=var_ranges)

evo = Evolution(problem)
final_pop = evo.evolve()


data = [{
    **{f"x{i}": val for i, val in enumerate(ind.features)},
    "F1": ind.objectives[0],
    "F2": ind.objectives[1]
} for ind in final_pop]

df = pd.DataFrame(data)
print(df.head())

# Optionnel : sauvegarde
df.to_csv("results_population.csv", index=False)

# Plot results
funcs = [ind.objectives for ind in final_pop]
f1_vals = [f[0] for f in funcs]
f2_vals = [f[1] for f in funcs]

plt.xlabel("F1 (cost)")
plt.ylabel("F2 (energy)")
plt.scatter(f1_vals, f2_vals)
plt.title("Pareto Front (NSGA-II)")
plt.show()
