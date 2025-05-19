import os
import json
from pathlib import Path

IoU_dir = "/work/kawano/LA/yolo_research/IoU_african-wildlife_noise"
quority_path = "/work/kawano/LA/yolo_research/quority_african-wildlife_noise/african_wildlife_elephant_IoU.json"
#quority_path = "/work/kawano/LA/yolo_research/quority_african-wildlife_noise/african_wildlife_elephant_objectlab.json"

with open(quority_path, 'r') as f:
    quority = json.load(f)

IoUs = []
scores = []
for file_name in os.listdir(IoU_dir):
    file_path = os.path.join(IoU_dir, file_name)
    file = Path(file_name).stem

    with open(file_path, 'r') as f:
        lines = f.readlines()
        IoUs += [float(line.strip()) for line in lines]
    # badloc = quority[file]["badloc_scores"]
    # swapped = quority[file]["swapped_scores"]
    #overlooked = quority[file]["overlooked_scores"]
    #scores += badloc
    #scores += [badloc[i] * swapped[i] for i in range(len(badloc))]
    scores += [quority[file]["IoUs"][i] * quority[file]["confs"][i] for i in range(len(quority[file]["IoUs"]))]

import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.scatter(IoUs, scores, marker='o', linestyle='-', color='b')
ax.set_xlabel('IoU')
ax.set_ylabel('Score')
ax.set_title('IoU vs Score')
plt.grid()
plt.show()
plt.savefig("IoU_vs_Score_IoU.png")