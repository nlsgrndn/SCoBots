import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join("Pong.csv")
df = pd.read_csv(path)

path = os.path.join(".", "my_data", "Pong-v0", "space_like")
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(os.path.join(path, "train"))
    os.makedirs(os.path.join(path, "validation"))
    os.makedirs(os.path.join(path, "test"))
type_str = "train"
stack_idx = 0
for i, image_str in enumerate(df["OBS"]):
    if (i//20) % 2 == 1:
        continue
    if i ==320:
        type_str = "validation"
        stack_idx = 0
    if i == 360:
        type_str = "test"
        stack_idx = 0
    if i >=400:
        break
    image = np.array([int(x) for x in image_str[1:-1].split(",")], dtype= np.uint8).reshape(210,160,3)
    # store image as png file
    plt.imsave(os.path.join(path, type_str, f'{(stack_idx//4):05}_{i%4}.png'), image)
    stack_idx +=1
