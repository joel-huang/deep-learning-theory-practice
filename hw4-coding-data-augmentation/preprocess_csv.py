
import numpy as np
import pandas as pd
import getimagenetclasses as g

image_paths = []

# obtain image paths in 'paths.txt' by running:
# ---------------------------------------------
# ls data/imagespart | cat > paths.txt
# ---------------------------------------------
with open('paths.txt') as f:
    for line in f.readlines():
        image_paths.append(line.strip('\n'))

pairs = {}
        
for path in image_paths:
    pairs[path] = g.get_label(path)

df = pd.DataFrame.from_dict(pairs, orient='index')
df.to_csv('data.csv')