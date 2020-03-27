import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv("result.csv")
fig, ax = plt.subplots()

ax.plot(data["trial"], data["timesteps"])
plt.show()