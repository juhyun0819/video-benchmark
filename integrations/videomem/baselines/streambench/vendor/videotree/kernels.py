# Verbatim function definitions from Ziyang412/VideoTree, MIT licensed.
# Origin revision and per-function SHA256 are recorded in provenance.json.
# Only import wiring differs: optional SoftDTW is not used by cosine VideoTree.
from functools import partial
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster

def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed) ; indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=15,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    # Dictionary to store the indices of the closest points for each cluster
    closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
    
    # Iterate over each cluster
    for cluster_id in range(len(cluster_centers)):
        # Filter points belonging to the current cluster
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]
        
        # Calculate distances from points in the cluster to the cluster center
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:    
            
            # Find the index (within the cluster) of the point closest to the cluster center
            closest_idx_in_cluster = torch.argmin(distances).item()
            
            # Map back to the original index in x
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()
            
            # Store the global index
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)

    return closest_points_idx_per_cluster


def hierarchical_clustering_with_external_primary(video_features, cluster_ids, relevance_scores, num_subclusters=5, num_subsubclusters=5):
    clusters = {i: {} for i in range(0, max(cluster_ids)+1)}

    for cluster_id in set(cluster_ids):
        primary_indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]

        if cluster_id < len(relevance_scores):
            score = relevance_scores[cluster_id]
        else:
            score = 3

        if len(primary_indices) < 2:
            clusters[cluster_id] = primary_indices
            continue

        sub_features = video_features[primary_indices]

        if score == 1:
            clusters[cluster_id] = primary_indices
            continue

        linked_sub = linkage(sub_features, method='ward')
        sub_cluster_labels = fcluster(linked_sub, num_subclusters, criterion='maxclust')
        sub_cluster_labels = sub_cluster_labels - 1

        if score == 2:
            clusters[cluster_id] = {i: [primary_indices[j] for j in np.where(sub_cluster_labels == i)[0]] for i in range(0, num_subclusters)}
            continue

        for subcluster_id in range(0, num_subclusters):
            sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
            if len(sub_indices) < 2:
                continue

            subsub_features = sub_features[sub_indices]
            linked_subsub = linkage(subsub_features, method='ward')
            subsub_cluster_labels = fcluster(linked_subsub, num_subsubclusters, criterion='maxclust')
            subsub_cluster_labels = subsub_cluster_labels - 1

            clusters[cluster_id][subcluster_id] = {}
            for subsubcluster_id in range(0, num_subsubclusters):
                final_indices = sub_indices[np.where(subsub_cluster_labels == subsubcluster_id)[0]]  # Correctly index into sub_indices
                original_indices = [primary_indices[i] for i in final_indices]
                clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices

    return clusters


def cosine_similarity(points, centroid):
    """
    Calculate cosine similarity between points and centroid.
    Returns the cosine distances (1 - similarity).
    """
    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()


def find_closest_points_in_temporal_order_subsub(x, clusters, relevance_scores):
    closest_points_indices = []

    for cluster_id, cluster_data in clusters.items():
        # print("cluster_id",cluster_id)
        # print("cluster_data type",type(cluster_data))

        if cluster_id < len(relevance_scores):
            relevance = relevance_scores[cluster_id]
        else:
            relevance = 3

        if isinstance(cluster_data, list):  # Primary cluster directly
            cluster_data = np.array(cluster_data)
            if cluster_data.size == 0:
                continue  # Skip empty clusters
            points_in_cluster = x[torch.tensor(cluster_data, dtype=torch.long)]
            cluster_centroid = points_in_cluster.mean(dim=0)
            distances = cosine_similarity(points_in_cluster, cluster_centroid)
            if distances.numel() > 0:
                closest_idx = torch.argmin(distances).item()
                closest_points_indices.append(int(cluster_data[closest_idx]))

        elif isinstance(cluster_data, dict):  # Handle subclusters and sub-subclusters
            if relevance == 1:
                # Only take the representative frame for the primary cluster
                print("line 207")
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if sub_data.size > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        subcluster_data = np.array(subcluster_data)
                        primary_indices.append(subcluster_data)

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))
                continue

            elif relevance == 2 or relevance == 3:
                # Include primary cluster representative
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if len(sub_data) > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        subcluster_data = np.array(subcluster_data)
                        primary_indices.append(subcluster_data)

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))

                for subcluster_id, subclusters in cluster_data.items():
                    if isinstance(subclusters, dict):  # Sub-subclusters
                        for subsubcluster_id, indices in subclusters.items():
                            if len(indices) == 0:
                                continue  # Skip empty sub-subclusters
                            indices_tensor = torch.tensor(indices, dtype=torch.long)
                            points_in_subsubcluster = x[indices_tensor]
                            subsubcluster_centroid = points_in_subsubcluster.mean(dim=0)
                            distances = cosine_similarity(points_in_subsubcluster, subsubcluster_centroid)
                            if distances.numel() > 0:
                                closest_idx_in_subsubcluster = torch.argmin(distances).item()
                                closest_global_idx = indices[closest_idx_in_subsubcluster]
                                closest_points_indices.append(int(closest_global_idx))

                    elif isinstance(subclusters, list):
                        subclusters = np.array(subclusters)
                        if subclusters.size == 0:
                            continue  # Skip empty subclusters
                        points_in_subcluster = x[torch.tensor(subclusters, dtype=torch.long)]
                        subcluster_centroid = points_in_subcluster.mean(dim=0)
                        distances = cosine_similarity(points_in_subcluster, subcluster_centroid)
                        if distances.numel() > 0:
                            closest_idx = torch.argmin(distances).item()
                            closest_points_indices.append(int(subclusters[closest_idx]))

    closest_points_indices.sort()  # Ensure the points are in temporal order
    return closest_points_indices
