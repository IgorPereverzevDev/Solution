import csv

import pandas as pd
from sklearn.cluster import MeanShift

with open('checkins.dat') as dat_file, open('data.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)

    for line in dat_file:
        row = [field.strip() for field in line.split('|')]
        if len(row) == 6 and row[3] and row[4]:
            csv_writer.writerow(row)

data = pd.read_csv('data.csv')
data = data.drop("created_at", 1).drop("id", 1).drop("user_id", 1).drop("venue_id", 1)

data_sample = data.sample(100000)
mean_sh = MeanShift(bandwidth=0.1, min_bin_freq=16, n_jobs=4)
mean_sh.fit(data_sample)

data_sample['cluster'] = mean_sh.predict(data_sample)
cluster_size = pd.DataFrame(data_sample.pivot_table(index='cluster', aggfunc='count', values='latitude'))
cluster_size.columns = ['clust_size']
cluster_centers = pd.DataFrame(mean_sh.cluster_centers_)
cluster_centers.columns = ['cent_latitude', 'cent_longitude']
cluster_df = cluster_centers.join(cluster_size)
cluster_df = cluster_df[cluster_df['clust_size'] > 15]

office_coordinates = [
    (33.751277, -118.188740),
    (25.867736, -80.324116),
    (51.503016, -0.075479),
    (52.378894, 4.885084),
    (39.366487, 117.036146),
    (-33.868457, 151.205134)
]


def get_distance(lat1, lon1, lat2, lon2):
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def get_min_distance(lat, lon):
    min_dist = None
    for (of_lat, of_lon) in office_coordinates:
        dist = get_distance(lat, lon, of_lat, of_lon)
        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
    return min_dist


cluster_df['min_distance'] = list(map(get_min_distance, cluster_df.cent_latitude, cluster_df.cent_longitude))

cluster_df = cluster_df.sort_values('min_distance')[:20]

with open('submission-cluster.txt', 'a') as file_obj:
    file_obj.write(str(cluster_df['cent_latitude'].iloc[0]) + " " + str(cluster_df['cent_longitude'].iloc[0]))
