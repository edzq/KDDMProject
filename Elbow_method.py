from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np


#load the processing files
import pickle
with open("plot_data/X_reduced.p",'rb') as file_handle:
    X_reduced = pickle.load(file_handle)
with open("plot_data/X.p",'rb') as file_handle:
    X = pickle.load(file_handle)


# run kmeans with many different k
distortions = []
K = range(2, 50)
pbar = tqdm(total=len(K))
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42, n_jobs=-1).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #print('Found distortion for {} clusters'.format(k))
    pbar.update(1)
    
    
#save the processed files
pickle.dump(distortions, open("plot_data/distortions.p", "wb" ))

# X_line = [K[0], K[-1]]
# Y_line = [distortions[0], distortions[-1]]
