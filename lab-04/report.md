# Uczenie nienadzorowane - Raport

## Ćwiczenie 1
Poeksperymentuj z innymi typami zbiorów danych (patrz: księżyce i koła poniżej) i spróbuj określić, jaki typ algorytmu klasteryzacji sprawdzi się dla nich najlepiej. Pamiętaj o sprawdzeniu parametrów dla różnych algorytmów, np.:

* k dla KMeans,
* eps dla DBSCAN,
* distance_threshold, affinity lub linkage dla AgglomerativeClustering.

### Księżyce
#### Oryginał
```
X, y = make_moons(n_samples=200, noise=0.05)
show_scatter(X)
```
![1](moon1.png)

#### KMeans
```
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
print(kmeans.inertia_)
show_scatter(X, y_pred, centers)
```
![2](moon2.png)

#### Ustalamy liczbę klastrów

```
km_list = list()

for k in range(1,10):
    km = KMeans(n_clusters=k)
    y_pred = km.fit(X)
    km_list.append(pd.Series({'clusters': k, 
                              'inertia': km.inertia_}))
```

```
plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,10,1))
ax.set_xlim(0,10)
ax.set(xlabel='Cluster', ylabel='Inertia');
```
![3](moon3.png)

#### MeanShift
```
ms = MeanShift(cluster_all=False)
y_pred = ms.fit_predict(X)
centers = ms.cluster_centers_
print("Number of clusters: ", len(centers))
show_scatter(X, y_pred, centers)
```
![4](moon4.png)

#### MeanShift z `bandwidth`

```
bandwidth = estimate_bandwidth(X, quantile=.35, n_samples=200) 
ms = MeanShift(cluster_all=False, bandwidth=bandwidth)
y_pred = ms.fit_predict(X)
centers = ms.cluster_centers_
show_scatter(X, y_pred, centers)
```

Najlepszymi wynikami dla tego algorytmu obrazują następujące wykresy, odpowiednio dla `quantile` = 0.35\
![5](moon5.png)\
oraz `quantile` = 0.6\
![6](moon6.png)\
Przy czym wynik drugi wydaje mi się lepszy, ponieważ chociaż jedna grupa jest odpowiednio przyporządkowana, podczas gdy dla pierwszego przypadku obie grupy są wymieszane.

#### DBSCAN

```
dbscan = DBSCAN(eps=0.3)
y_pred = dbscan.fit_predict(X)
print('Number of clusters:', len(set(y_pred))-(1 if -1 in y_pred else 0))
print('Number of outliers:', list(y_pred).count(-1))
show_scatter(X, y_pred)
```
Zadowalający wynik uzyskałem dla `eps` = 0.3\
![7](moon7.png)

#### AgglomerativeClustering

```
ac = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, 
                             affinity='euclidean', linkage='single')
y_pred = ac.fit_predict(X)
print('Number of clusters:', len(set(y_pred)))
```
Zadowalające wyniki uzyskałem dla następujących parametrów
* `distance_threshold` = 0.2, `affinity` = 'euclidean', `linkage` = 'single'\
##### Wynik
![8](moon8.png)
##### Dendrogram
![9](moon9.png)

* `distance_threshold` = 0.3, `affinity` = 'l1', `linkage` = 'single'\
##### Wynik
![10](moon10.png)
##### Dendrogram
![11](moon11.png)

* `distance_threshold` = 0.2, `affinity` = 'l2', `linkage` = 'single'\
##### Wynik
![12](moon12.png)
##### Dendrogram
![13](moon13.png)

* `distance_threshold` = 0.25, `affinity` = 'manhattan', `linkage` = 'single'\
##### Wynik
![14](moon14.png)
##### Dendrogram
![15](moon15.png)

### Koła

#### Oryginał
```
X, y = make_circles(n_samples=200, factor=0.5, noise=0.05)
show_scatter(X)
```
![1](circle1.png)

#### KMeans
```
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
print(kmeans.inertia_)
show_scatter(X, y_pred, centers)
```
![2](circle2.png)

#### Ustalamy liczbę klastrów

```
km_list = list()

for k in range(1,10):
    km = KMeans(n_clusters=k)
    y_pred = km.fit(X)
    km_list.append(pd.Series({'clusters': k, 
                              'inertia': km.inertia_}))
```

```
plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,10,1))
ax.set_xlim(0,10)
ax.set(xlabel='Cluster', ylabel='Inertia');
```
![3](circle3.png)

#### MeanShift
```
ms = MeanShift(cluster_all=False)
y_pred = ms.fit_predict(X)
centers = ms.cluster_centers_
print("Number of clusters: ", len(centers))
show_scatter(X, y_pred, centers)
```
![4](circle4.png)
