import pandas as pd
import networkx
import matplotlib.pyplot as plt

oldRoute = pd.read_csv("./data3.csv", encoding="gb2312")
newRoute = pd.read_csv("./data4.csv", encoding="gb2312")

oldRoutes = []
newRoutes = []
for start, end, _ in oldRoute.values:
    oldRoutes.append((start, end))

for start, end in newRoute.values:
    newRoutes.append((start, end))

graph = networkx.Graph()
graph.add_edges_from(oldRoutes)
networkx.draw(graph, with_labels=True)
plt.show()