
import pandas as pd

H, D, K, N = 3, 2, 2, 4 # tes tailles
df = pd.read_csv("results_population.csv")

def decode_solution(row):
    x = [row[f"x{i}"] for i in range(H*D + H*K + N*H + N*H)]

    a = x[0:H*D]
    xhk = x[H*D:H*D + H*K]
    p = x[H*D + H*K:H*D + H*K + N*H]
    z = x[H*D + H*K + N*H:H*D + H*K + 2*N*H]

    return {
        "a": [round(v) for v in a],
        "x": [round(v) for v in xhk],
        "p": [round(v) for v in p],
        "z": [round(v, 2) for v in z],
        "F1": row["F1"],
        "F2": row["F2"]
    }

decoded = df.apply(decode_solution, axis=1)
decoded.to_csv("decode_results_population.csv", index=False)
for i, sol in enumerate(decoded[:5]):  # affiche les 5 premières
    print(f"Solution {i+1} :", sol)
    print("-" * 50)
