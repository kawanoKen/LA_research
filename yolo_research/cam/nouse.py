import json
import numpy as np

n = 200
arr = np.random.rand(n, 100).astype(np.float32)  # 例   shape=(100,n)

# ⚠ json は NumPy を直接シリアライズできない
data = arr.tolist()               # ndarray → list of list

with open("array.json", "w") as f:
    for row in arr:                 # (n,) → list
        json.dump(row.tolist(), f)
        f.write("\n")