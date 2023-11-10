import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import hdbscan


os.environ["OMP_NUM_THREADS"] = '4'
path = '../artificial/'

def Load_initial_data(name) : 
    #path_out = './fig/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])

    # PLOT datanp (en 2D) - / scatter plot
    # Extraire chaque valeur de features pour en faire une liste
    # EX : 
    # - pour t1=t[:,0] --> [1, 3, 5, 7]
    # - pour t2=t[:,1] --> [2, 4, 6, 8]
    print("---------------------------------------")
    print("Affichage données initiales            "+ str(name))
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne

    #plt.figure(figsize=(6, 6))
    plt.scatter(f0, f1, s=8)
    plt.title("Donnees initiales : "+ str(name))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()
    loaded_data_result = {}
    loaded_data_result ["data"] = datanp
    loaded_data_result ["f0"] = f0
    loaded_data_result ["f1"] = f1
    return loaded_data_result

def dbscan_iteration(loaded_data, epsilon, min_pts): 
    tps1 = time.time()
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(loaded_data["data"])
    tps2 = time.time()
    labels = model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    
    plt.scatter(loaded_data["f0"],loaded_data["f1"],c = labels,s = 8 )
    plt.title("DBSCAN clustering | Epsilon= "+str(epsilon)+" | Min_samples= "+str(min_pts) + " | k=" + str(n_clusters))
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig("pmf_dbscan.png", format = 'jpg')
    plt.show()
    dbscan_result = {}
    dbscan_result["runtime"]=round (( tps2 - tps1 )*1000, 2 )
    dbscan_result["k"]=n_clusters
    dbscan_result["labels"]=labels
    return dbscan_result

def standardized_data(loaded_data) : 
    scaler = preprocessing.StandardScaler().fit(loaded_data["data"])
    data_scaled = scaler.transform(loaded_data["data"])
    print("Affichage données standardisées            ")
    f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
    f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

    plt.scatter(f0_scaled, f1_scaled, s=8)
    plt.title("Donnees standardisées")
    plt.show()
    data_scaled_result = {}
    data_scaled_result ["data"] = data_scaled
    data_scaled_result ["f0"] = f0_scaled
    data_scaled_result ["f1"] = f1_scaled
    return data_scaled_result

def neighbors_eps(data_scaled, k) : 
    neigh = NearestNeighbors(n_neighbors=k) 
    neigh.fit(data_scaled["data"])
    distances , indices = neigh.kneighbors(data_scaled["data"])
    #distance moyenne sur les k plus proches voisins en retirant le point "origine"
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0,distances.shape[0])])
    #trier par ordre croissant
    distancetrie = np.sort(newDistances)
    plt.title("Méthode du Coude - Variation des Distances Moyennes aux k Plus Proches Voisins " +str(k) )
    plt.xlabel("Nombre de Voisins (k)")
    plt.ylabel("Distance Moyenne")
    plt.plot(distancetrie)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig("Methode_du_coude_plot.png", format = 'jpg')
    plt.show()
    return max(distancetrie)

def hdbscan_iteration(loaded_data, min_samples, min_cluster_size): 
    tps1 = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples)  
    clusterer.fit(loaded_data["data"])
    tps2 = time.time()
    labels = clusterer.labels_

    # Number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    # Visualize the clustering results
    plt.scatter(loaded_data["f0"],loaded_data["f1"],c = labels,s = 8 )
    plt.title("HDBSCAN Clustering | Min_samples = "+ str(min_samples)+ " | Min_cluster_size = "+ str(min_cluster_size) + "| k = "+str(n_clusters))
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig("pmf_hdbscan.png", format = 'jpg')
    plt.show()
    print("For min_Estimated = %d, number of clusters: %d" % (min_samples,n_clusters))
    hdbscan_result = {}
    hdbscan_result["runtime"] = round((tps2 - tps1)*1000,2)
    hdbscan_result["k"] = n_clusters
    hdbscan_result["labels"] = labels 
    
    sil_score = metrics.silhouette_score(loaded_data["data"], labels, metric = 'euclidean')
    print("The silhouette score is " + str(sil_score))
    
    return hdbscan_result

#Function that determines the best parameteres for dbscan
def dbscan_find_best_params(loaded_data, max_min_samples):
    k_range = []
    sil_score = []
    dav_score = []
    runtime_list = []
    best_silhouette = -1
    best_min_samples = 0
    best_eps = 0 
    best_n_clusters = 0 
    
    for i in range(2, max_min_samples): 
        eps = neighbors_eps(loaded_data, i) *0.2
        dbscan_return = dbscan_iteration(loaded_data, eps,i)
        if dbscan_return["k"] > 1 : 
            k_range.append(i)
            runtime_list.append(dbscan_return["runtime"]*0.01)
            sil_sc= metrics.silhouette_score(loaded_data["data"], dbscan_return["labels"], metric = 'euclidean')
            sil_score.append(sil_sc)
            dav_sc = metrics.davies_bouldin_score(loaded_data["data"], dbscan_return["labels"])
            if(sil_sc > best_silhouette): 
                best_silhouette = sil_sc 
                best_min_samples = i 
                best_eps = eps 
                best_n_clusters = dbscan_return["k"]
    plt.plot(k_range, runtime_list, label = 'Runtime (10**-1 s)')
    plt.title("Evaluation selon le nombre de cluster")
    plt.legend()
    print("Best silhouette score = ", best_silhouette, " | Best min_samples ", best_min_samples," |  Best epsilon : ", best_eps, " | Number of clusters : ", best_n_clusters)
    return [best_eps, best_min_samples, best_silhouette, runtime_list, k_range]

#Function that determines the best parameteres for hdbscan
def hdbscan_find_best_params(loaded_data, max_min_samples,min_cluster_size_range):
    k_range = []
    sil_score = []
    dav_score = []
    runtime_list = []
    best_silhouette = -1
    best_min_samples = 0
    best_min_cluster_size = 0
    best_n_clusters = 0 
    for i in range(2, max_min_samples): 
        for min_cluster_size in min_cluster_size_range:
            hdbscan_return = hdbscan_iteration(loaded_data,i,min_cluster_size)
            k_range.append(i)
            runtime_list.append(hdbscan_return["runtime"]*0.01)
            sil_sc= metrics.silhouette_score(loaded_data["data"], hdbscan_return["labels"], metric = 'euclidean')
            sil_score.append(sil_sc)
            dav_sc = metrics.davies_bouldin_score(loaded_data["data"], hdbscan_return["labels"])
            if(sil_sc > best_silhouette): 
                best_silhouette = sil_sc 
                best_min_samples = i 
                best_min_cluster_size = min_cluster_size
                best_n_clusters = hdbscan_return["k"]
                
    #plt.plot(k_range, runtime_list, label = 'Runtime (10**-1 s)')
    #plt.title("Evaluation selon le nombre de cluster et le min_cluster_size")
    #plt.legend()
    print("Best silhouette score =", best_silhouette, "Best min_samples", best_min_samples, ", Best min_cluster_size : ", best_min_cluster_size, ", Number of clusters : ", best_n_clusters)
    return [best_min_samples, best_silhouette, runtime_list, k_range]