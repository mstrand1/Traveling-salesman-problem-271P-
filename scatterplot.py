import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go

x1 = np.random.normal(.5, 50, 500)
y1 = np.random.normal(150, 50, 500)
z1 = np.random.normal(1, 50, 500)

DF = np.array((x1, y1, z1), dtype=float)
DF1 = pd.DataFrame(DF.T)

DF1['cluster'] = 0
DF1.columns =['data1', 'data2', 'data3','cluster']

# Creating our Model
kmeans = KMeans(n_clusters = 3)

# Training our model
kmeans.fit(DF1)

# You can see the labels (clusters) assigned for each data point with the function labels_
kmeans.labels_

# Assigning the labels to the initial dataset
DF1['cluster'] = kmeans.labels_

PLOT = go.Figure()

for C in list(DF1.cluster.unique()):
    PLOT.add_trace(go.Scatter3d(x=DF1[DF1.cluster == C]['data1'],
                                y=DF1[DF1.cluster == C]['data2'],
                                z=DF1[DF1.cluster == C]['data3'],
                                mode='markers', marker_size=8, marker_line_width=1,
                                name='Cluster ' + str(C)))

PLOT.update_layout(width=800, title='Clustering with KMeans where number of clusters is 3', height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='Norm 1', titlefont_color='black'),
                              yaxis=dict(title='Norm 2', titlefont_color='black'),
                              zaxis=dict(title='Norm 3', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))

PLOT.show()
