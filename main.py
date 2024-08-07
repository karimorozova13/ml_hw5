# %%
import pickle

with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# %%
concrete = datasets['concrete']

# %%
components = ['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']

concrete['Components'] = concrete[components].gt(0).sum(axis=1)

concrete[components + ['Components']].head(10)

# %%
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(concrete)

# %%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

model_kmn = KMeans(random_state=42)

visualizer = KElbowVisualizer(
    model_kmn,
    k=(2,10),
    timings=False
    )

visualizer.fit(X)
visualizer.show()

# %%
import pandas as pd

k_best = visualizer.elbow_value_

model_kmn  = KMeans(n_clusters=k_best, random_state=42).fit(X)


labels_kmn = pd.Series(model_kmn.labels_, name='k-means')

concrete['Cluster'] = labels_kmn

# %%
report = concrete.groupby('Cluster').median()

# %%
cluster_counts = concrete['Cluster'].value_counts().sort_index()
report['Count'] = cluster_counts

report.head()

# %%
print(report.to_string())
print("\n" + "-"*40 + "\n")

for i, row in report.iterrows():
    print(f'Cluster {i}:')
    print(row.to_string())
    print("\n" + "-"*40 + "\n")







