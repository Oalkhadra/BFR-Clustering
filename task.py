import sys
import time
import numpy as np
from sklearn.cluster import KMeans

def Mahalanobis(point, centroid_stats):
    # Get centroid and std_dev from cluster statistics
    centroid = centroid_stats['centroid']
    std_dev = centroid_stats['std_dev']

    # For points where std_dev is 0, set it to a small number to avoid division by zero
    std_dev = np.where(std_dev == 0, 1e-10, std_dev)

    # Calculate normalized distance
    normalized_diff = (point - centroid) / std_dev

    # Return Euclidean norm of normalized differences
    return np.sqrt(np.sum(normalized_diff ** 2))

def BFR_Initial(data, input_n_cluster):
    # Initialize empty sets: DS, CS, and RS
    DS, CS, RS = set(), set(), set()

    # Step 1: Shuffle and split data into chunks of 20%, use first chunk
    np.random.shuffle(data)
    chunks = np.array_split(data, 5)
    chunk_1 = chunks[0]
    first_chunk_features = chunk_1[:, 2:]

    # Step 2: Run K-means with large K on first chunk
    kmeans = KMeans(n_clusters=input_n_cluster * 5)
    initial_cluster_labels = kmeans.fit_predict(first_chunk_features)

    # Step 3: Move all clusters with only 1 point to RS
    cluster_sizes = np.bincount(initial_cluster_labels)
    single_point_clusters = set(np.where(cluster_sizes == 1)[0])

    for idx, (point, cluster_label) in enumerate(zip(chunk_1, initial_cluster_labels)):
        point_index = int(point[0])

        if cluster_label in single_point_clusters:
            # Add to RS if point is in a single-point cluster
            RS.add(point_index)

    # Step 4: Run K-means on rest of chunk, using k = input_clusters
    non_rs_mask = np.array([cluster_label not in single_point_clusters
                            for cluster_label in initial_cluster_labels])

    non_rs_points = chunk_1[non_rs_mask]
    non_rs_features = first_chunk_features[non_rs_mask]

    kmeans = KMeans(n_clusters=input_n_cluster)
    step4_cluster_labels = kmeans.fit_predict(non_rs_features)

    for point, cluster_label in zip(non_rs_points, step4_cluster_labels):
        point_index = int(point[0])
        DS.add(point_index)

    # Step 5: Store DS statistics (Also initialize CS_statistics)
    DS_statistics = {}
    CS_statistics = {}

    for cluster_id in range(input_n_cluster):
        # Get points belonging to current cluster
        cluster_mask = step4_cluster_labels == cluster_id
        cluster_points = non_rs_features[cluster_mask]

        # Calculate N, SUM, and SUMSQ
        N = len(cluster_points)
        SUM = np.sum(cluster_points, axis=0)
        SUMSQ = np.sum(cluster_points ** 2, axis=0)

        # Store statistics
        DS_statistics[cluster_id] = {
            'N': N,
            'SUM': SUM,
            'SUMSQ': SUMSQ,
            'centroid': SUM / N,
            'std_dev': np.sqrt((SUMSQ - (SUM ** 2) / N) / N)
        }

    # Step 6: Run K-means on points in RS, generate CS and RS
    if len(RS) >= (input_n_cluster * 3):
        # Get features of RS points
        rs_points_mask = np.array([int(point[0]) in RS for point in chunk_1])
        rs_points = chunk_1[rs_points_mask]
        rs_features = first_chunk_features[rs_points_mask]

        # Run k-means on RS points, count cluster sizes
        rs_kmeans = KMeans(n_clusters=input_n_cluster * 3)
        rs_cluster_labels = rs_kmeans.fit_predict(rs_features)
        rs_cluster_sizes = np.bincount(rs_cluster_labels)

        # Move clusters with > 1 point to CS, keep others in RS; Generate CS Statistics
        RS.clear()

        for cluster_id in range(len(rs_cluster_sizes)):
            cluster_mask = rs_cluster_labels == cluster_id
            cluster_points = rs_features[cluster_mask]

            if len(cluster_points) > 1:  # Only process clusters with more than 1 point
                # Calculate statistics for this CS cluster
                N = len(cluster_points)
                SUM = np.sum(cluster_points, axis=0)
                SUMSQ = np.sum(cluster_points ** 2, axis=0)

                CS_statistics[cluster_id] = {
                    'N': N,
                    'SUM': SUM,
                    'SUMSQ': SUMSQ,
                    'centroid': SUM / N,
                    'std_dev': np.sqrt((SUMSQ - (SUM ** 2) / N) / N)
                }

                # Add points to CS
                for point in rs_points[cluster_mask]:
                    point_index = int(point[0])
                    CS.add(point_index)
            else:
                # Add single points to RS
                for point in rs_points[cluster_mask]:
                    point_index = int(point[0])
                    RS.add(point_index)

    with open(output_filepath, 'w') as f:
        f.write(f"The intermediate results:\n")
        f.write(f"Round 1: {len(DS)},{len(CS_statistics)},{len(CS)},{len(RS)}\n")

    return chunks, DS_statistics, CS_statistics, DS, RS, CS

# Define helper function to merge CS's
def mergeCS(cs_stats, threshold):
    merged = False
    cs_ids = list(cs_stats.keys())

    for i in range(len(cs_ids)):
        if cs_ids[i] not in cs_stats:  # Skip if already merged
            continue

        for j in range(i + 1, len(cs_ids)):
            if cs_ids[j] not in cs_stats:  # Skip if already merged
                continue

            cluster1 = cs_stats[cs_ids[i]]
            cluster2 = cs_stats[cs_ids[j]]

            # Calculate Mahalanobis distance between centroids
            centroid1 = cluster1['centroid']
            centroid2 = cluster2['centroid']
            avg_std_dev = (cluster1['std_dev'] + cluster2['std_dev']) / 2

            # Use average std_dev of both clusters
            normalized_diff = (centroid1 - centroid2) / avg_std_dev
            distance = np.sqrt(np.sum(normalized_diff ** 2))

            # If clusters are close enough, merge them
            if distance <= threshold:
                # Merge statistics
                new_N = cluster1['N'] + cluster2['N']
                new_SUM = cluster1['SUM'] + cluster2['SUM']
                new_SUMSQ = cluster1['SUMSQ'] + cluster2['SUMSQ']

                # Update first cluster with merged statistics
                cs_stats[cs_ids[i]] = {
                    'N': new_N,
                    'SUM': new_SUM,
                    'SUMSQ': new_SUMSQ,
                    'centroid': new_SUM / new_N,
                    'std_dev': np.sqrt((new_SUMSQ - (new_SUM ** 2) / new_N) / new_N)
                }

                # Remove second cluster
                del cs_stats[cs_ids[j]]
                merged = True

        return merged

# Helper function to merge CS with DS
def merge_cs_to_ds_clusters(cs_stats, ds_stats, CS, DS, threshold):
    merged = False
    cs_ids = list(cs_stats.keys())

    for cs_id in cs_ids:
        if cs_id not in cs_stats:  # Skip if already merged
            continue

        min_distance = float('inf')
        best_ds_cluster = None
        cs_cluster = cs_stats[cs_id]

        # Find the closest DS cluster
        for ds_id, ds_cluster in ds_stats.items():
            # Calculate Mahalanobis distance between centroids
            cs_centroid = cs_cluster['centroid']
            ds_centroid = ds_cluster['centroid']
            avg_std_dev = (cs_cluster['std_dev'] + ds_cluster['std_dev']) / 2

            # Use average std_dev of both clusters
            normalized_diff = (cs_centroid - ds_centroid) / avg_std_dev
            distance = np.sqrt(np.sum(normalized_diff ** 2))

            if distance < min_distance:
                min_distance = distance
                best_ds_cluster = ds_id

        # If closest DS cluster is within threshold, merge
        if min_distance <= threshold:
            ds_cluster = ds_stats[best_ds_cluster]
            new_N = ds_cluster['N'] + cs_cluster['N']
            new_SUM = ds_cluster['SUM'] + cs_cluster['SUM']
            new_SUMSQ = ds_cluster['SUMSQ'] + cs_cluster['SUMSQ']

            # Update DS cluster statistics
            ds_stats[best_ds_cluster] = {
                'N': new_N,
                'SUM': new_SUM,
                'SUMSQ': new_SUMSQ,
                'centroid': new_SUM / new_N,
                'std_dev': np.sqrt((new_SUMSQ - (new_SUM ** 2) / new_N) / new_N)
            }

            # Move all points from CS to DS
            CS_points = {int(point[0]) for point in data if int(point[0]) in CS}
            DS.update(CS_points)
            CS.difference_update(CS_points)

            # Remove merged CS cluster
            del cs_stats[cs_id]
            merged = True

    return merged

# Define function for rounds 2-5
def BFR_Remainder(chunks, initial_DS_stats, initial_CS_stats, DS, RS, CS, input_n_cluster):
    # Attain inputs
    chunks = chunks
    ds_stats, cs_stats = initial_DS_stats, initial_CS_stats
    DS, RS, CS = DS, RS, CS

    # Loop through each of the remaining chunks
    for i in range(1, 5):
        current_chunk = chunks[i]
        current_chunk_features = current_chunk[:, 2:]

        # Steps 8-10: merge points with current sets
        threshold = 2 * np.sqrt(current_chunk_features.shape[1])

        # For each point in the chunk
        for idx, point in enumerate(current_chunk):
            point_index = int(point[0])
            point_features = current_chunk_features[idx]

            min_distance = float('inf')
            best_cluster = None

            # Compare with each DS cluster
            for cluster_id, cluster_stats in ds_stats.items():
                distance = Mahalanobis(point_features, cluster_stats)

                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_id

            # Assign point based on distance threshold
            if min_distance <= threshold:
                # Add point to nearest DS cluster
                DS.add(point_index)

                # Update statistics for the assigned cluster
                cluster = ds_stats[best_cluster]
                cluster['N'] += 1
                cluster['SUM'] += point_features
                cluster['SUMSQ'] += point_features ** 2
                cluster['centroid'] = cluster['SUM'] / cluster['N']
                cluster['std_dev'] = np.sqrt((cluster['SUMSQ'] -
                                              (cluster['SUM'] ** 2) / cluster['N']) /
                                             cluster['N'])

            # If point is not assigned to DS cluster, check distances with CS clusters
            else:
                # Compare with each CS cluster
                for cluster_id, cluster_stats in cs_stats.items():
                    distance = Mahalanobis(point_features, cluster_stats)

                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id

                # Assign point based on distance threshold
                if min_distance <= threshold:
                    # Add point to nearest CS cluster
                    CS.add(point_index)

                    # Update statistics for the assigned cluster
                    cluster = cs_stats[best_cluster]
                    cluster['N'] += 1
                    cluster['SUM'] += point_features
                    cluster['SUMSQ'] += point_features ** 2
                    cluster['centroid'] = cluster['SUM'] / cluster['N']
                    cluster['std_dev'] = np.sqrt((cluster['SUMSQ'] -
                                                  (cluster['SUM'] ** 2) / cluster['N']) /
                                                 cluster['N'])
                # If point is not assigned to DS or CS, add to RS
                else:
                    RS.add(point_index)

        # Step 11: Run K-means with larger K on RS to generate more CS clusters and RS points
        if len(RS) >= (input_n_cluster * 3):
            # Get features of RS points
            rs_points_mask = np.array([int(point[0]) in RS for point in data])
            rs_points = data[rs_points_mask]
            rs_features = rs_points[:, 2:]

            # Run k-means on RS points, count cluster sizes
            rs_kmeans = KMeans(n_clusters=input_n_cluster * 3)
            rs_cluster_labels = rs_kmeans.fit_predict(rs_features)
            rs_cluster_sizes = np.bincount(rs_cluster_labels)

            # Move clusters with > 1 point to CS, keep others in RS; Generate CS Statistics
            RS.clear()

            # Get the next available CS cluster ID
            next_cs_cluster_id = max(cs_stats.keys(), default=-1) + 1

            for cluster_id in range(len(rs_cluster_sizes)):
                cluster_mask = rs_cluster_labels == cluster_id
                cluster_points = rs_features[cluster_mask]

                if len(cluster_points) > 1:  # Only process clusters with more than 1 point
                    # Calculate statistics for this CS cluster
                    N = len(cluster_points)
                    SUM = np.sum(cluster_points, axis=0)
                    SUMSQ = np.sum(cluster_points ** 2, axis=0)

                    cs_stats[next_cs_cluster_id] = {
                        'N': N,
                        'SUM': SUM,
                        'SUMSQ': SUMSQ,
                        'centroid': SUM / N,
                        'std_dev': np.sqrt((SUMSQ - (SUM ** 2) / N) / N)
                    }

                    # Add points to CS
                    for point in rs_points[cluster_mask]:
                        point_index = int(point[0])
                        CS.add(point_index)

                    next_cs_cluster_id += 1
                else:
                    # Add single points to RS
                    for point in rs_points[cluster_mask]:
                        point_index = int(point[0])
                        RS.add(point_index)

        # Step 12: Merge CS clusters that have a Mahalanobis distance < 2root(d)
        if i < 4:
            while mergeCS(cs_stats, threshold):
                pass

        # If final round, merge potential CS with DS clusters, using threshold
        else:
            while merge_cs_to_ds_clusters(cs_stats, ds_stats, CS, DS, threshold):
                pass

        with open(output_filepath, 'a') as f:
            f.write(f"Round {i + 1}: {len(DS)},{len(cs_stats)},{len(CS)},{len(RS)}\n")

    return DS, CS, RS, ds_stats

# Helper function to identify point-cluster assignments
def getClusterAssignments(data, ds_stats):
    features = data[:, 2:]
    point_indices = data[:, 0].astype(int)

    # Initialize array to store minimum distances and cluster assignments
    min_distances = np.full(len(data), np.inf)
    assignments = np.full(len(data), -1)

    # Process each cluster
    for cluster_id, stats in ds_stats.items():
        centroid = stats['centroid']
        std_dev = np.where(stats['std_dev'] == 0, 1e-10, stats['std_dev'])

        # Vectorized Mahalanobis distance calculation for all points
        normalized_diff = (features - centroid) / std_dev
        distances = np.sqrt(np.sum(normalized_diff ** 2, axis=1))

        # Update assignments where this cluster is closer
        mask = distances < min_distances
        min_distances[mask] = distances[mask]
        assignments[mask] = cluster_id

    # Create dictionary mapping point indices to cluster assignments
    return dict(zip(point_indices, assignments))

if __name__ == '__main__':
    # Start timer
    timeStart = time.time()

    # Read inputs
    input_filepath = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_filepath = sys.argv[3]

    # Read dataset
    indices = []
    labels = []
    features = []

    with open(input_filepath, 'r') as f:
        for line in f:
            tok = line.strip().split(',')
            # Extract index and label
            indices.append(int(tok[0]))
            labels.append(int(tok[1]))

            # Convert remaining values to float and add to features
            row_features = [float(x) for x in tok[2:]]
            features.append(row_features)

    data = np.column_stack((np.array(indices).reshape(-1, 1),
                            np.array(labels).reshape(-1, 1),
                            np.array(features)))

    # Run BFR Algorithm
    chunks, DS_statistics, CS_statistics, DS, RS, CS = BFR_Initial(data, n_cluster)
    finalDS, fincalCS, finalRS, final_ds_stats = BFR_Remainder(chunks, DS_statistics, CS_statistics, DS, RS, CS, n_cluster)

    # Get final cluster assignments
    cluster_assignments = getClusterAssignments(data, final_ds_stats)

    # Write remainder of output
    with open(output_filepath, 'a') as f:
        f.write("\nThe clustering results:\n")
        # For each point in the dataset
        for idx in range(len(data)):
            # If point is in finalDS, use its cluster assignment, otherwise use -1
            cluster = cluster_assignments.get(idx, -1) if idx in finalDS else -1
            f.write(f"{idx},{cluster}\n")

    print(f"Duration: {time.time() - timeStart}")