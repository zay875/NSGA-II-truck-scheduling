
import pandas as pd

H, D, K, N = 5, 2, 2, 4
df = pd.read_csv("results_population_truck_det_constraint.csv")

def decode_solution(row):
    x = [row[f"x{i}"] for i in range(H*D + H*K + H + N*H + N*H)]

    a = x[0:H*D]
    xhk = x[H*D:H*D + H*K]
    v_h = x[H*D + H*K:H*D + H*K + H]
    p = x[H*D + H*K + H:H*D + H*K + H + N*H]
    z = x[H*D + H*K + H + N*H:H*D + H*K  + H + 2*N*H]

    return {
        "a": [1 if v > 0.5 else 0 for v in a],
        "x": [1 if v > 0.5 else 0 for v in xhk],
        "v" : [1 if v > 0.5 else 0 for v in v_h],
        "p": [1 if v > 0.5 else 0 for v in p],
        "z": [round(v, 2) for v in z],
        "F1": row["F1"],
        "F2": row["F2"]
    }

decoded = df.apply(decode_solution, axis=1)
decoded.to_csv("decode_results_population_truck_dest_constraint.csv", index=False)
for i, sol in enumerate(decoded[:5]):  # affiche les 5 premières
    print(f"Solution {i+1} :", sol)
    print("-" * 50)
