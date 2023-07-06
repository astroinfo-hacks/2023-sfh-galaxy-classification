
import pandas as pd

import numpy as np
import numpy
from sklearn.mixture import GaussianMixture

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import matplotlib.cm as cm


# In[2]:


# file name = binned_SFHs-7levels-JWST_z_0.5-1.0
file = "data/binned_SFHs-7levels-JWST_z_0.5-1.0.txt"
df = pd.read_csv(file,sep='\t')
df
levels=df.columns[2:8]
SFH_lev=df[levels].values


# In[3]:



seed = 0
numpy.random.seed(seed)
X_train = SFH_lev
#X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# Make time series shorter
sz = X_train.shape[1]


# In[4]:


# shuffle and reduce X_train by half
X_train = SFH_lev
numpy.random.shuffle(X_train)
#X_train = X_train[:len(X_train)//2]

X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
print(X_train.shape)

# Reshape X_train to 2 dimensions
nSamples, nx, ny = X_train.shape
X_train = X_train.reshape((nSamples, nx*ny))



# In[14]:


# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed, n_jobs=-1)
y_pred = km.fit_predict(X_train)

plt.figure(figsize=(12, 18), dpi=150)  # Increase the figure size and dpi

for yi in range(6):
    plt.subplot(6, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

plt.tight_layout()
plt.show()



# In[18]:



# project to 2D for plotting
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

unique_labels = np.unique(y_pred)
num_labels = len(unique_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))
cmap = ListedColormap(colors)

# Increase the resolution
plt.figure(figsize=(10, 8), dpi=500)

# plot data
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_pred, cmap=cmap, s=0.2)

# Set plot title and colorbar
plt.title('Euclidean k-means 6 CR, PCA projection')
plt.colorbar()

# Display the plot
plt.show()


# project to 3D for plotting
pca = PCA(n_components=3)
X_train_3d = pca.fit_transform(SFH_lev)

unique_labels = np.unique(y_pred)
num_labels = len(unique_labels)
colors = cm.rainbow(np.linspace(0, 1, num_labels))
cmap = ListedColormap(colors)

# Increase the resolution
fig = plt.figure(figsize=(10, 8), dpi=500)
ax = fig.add_subplot(111, projection='3d')

# plot data
scatter = ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c=y_pred, cmap=cmap, s=0.2)

# Set plot title
ax.set_title('Euclidean k-means 6 CR, PCA projection')

# Add colorbar
cbar = fig.colorbar(scatter)

# Display the plot
plt.show()


import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

# project to 3D for plotting
pca = PCA(n_components=3)
X_train_3d = pca.fit_transform(SFH_lev)

# Select the cluster you want to plot
cluster_to_plot = 0

# Filter the data points belonging to the selected cluster
X_cluster = X_train_3d

# Create a scatter plot trace for the selected cluster
scatter = go.Scatter3d(
    x=X_cluster[:, 0],
    y=X_cluster[:, 1],
    z=X_cluster[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=y_pred,
        colorscale='Rainbow',
        opacity=1
    )
)

# Create the layout
layout = go.Layout(
    title='Euclidean k-means 6 CR, PCA projection',
    scene=dict(
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        zaxis=dict(title='Component 3')
    ),
    showlegend=False
)

# Create the figure and add the scatter plot trace
fig = go.Figure(data=[scatter], layout=layout)

fig.show()


