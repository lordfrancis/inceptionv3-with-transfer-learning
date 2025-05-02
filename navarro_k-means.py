from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import mysql.connector

con = mysql.connector.connect(
    host ='129.146.175.178',
    user ='dit_816_user',
    password = 'AbcDD@12345678K',
    database = 'dit_816'
)

query = "SELECT * FROM grade_salary"
dataset = pd.read_sql(query, con)
# print(dataset.head())

# plt.scatter(dataset['grade'], dataset['salary'])
# plt.show()

#Check number of K
k_range = range(1, 10)
sse = []
for k in k_range:
    test_k = KMeans(n_clusters=k, n_init=10)
    test_k.fit(dataset[['grade', 'salary']])
    sse.append(test_k.inertia_)

# plt.plot(k_range, sse)
# plt.show()  #Results show that 3 or 4 would be best

#Kmeans clustering where K = 3
CLUSTERS = 4
km = KMeans(n_clusters=CLUSTERS, n_init=10)
km.fit(dataset[['grade', 'salary']])
dataset['cluster'] = km.predict(dataset[['grade', 'salary']])
plt.scatter(dataset.grade, dataset.salary, c='black', label='Data Points') #Check the data points

plt.show()

d1 = dataset[dataset.cluster == 0]
d2 = dataset[dataset.cluster == 1]
d3 = dataset[dataset.cluster == 2]
d4 = dataset[dataset.cluster == 3]



c1 = plt.scatter(d1.grade, d1.salary, c='red', label='Cluster 0')
c2 = plt.scatter(d2.grade, d2.salary, c='green', label='Cluster 1')
c3 = plt.scatter(d3.grade, d3.salary, c='blue', label='Cluster 2')
c4 = plt.scatter(d4.grade, d4.salary, c='yellow', label='Cluster 3')
plt.legend((c1,c2,c3,c4), ('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'))
plt.xlabel('Grade')
plt.ylabel('Salary')
plt.title('KMeans Clustering')
# plt.show()

#Test dataset 1
test_grade1 = 4.5
test_salary1 = 3514
test_dataset1 = pd.DataFrame({'grade': [test_grade1], 'salary': [test_salary1]})
test_cluster1 = km.predict(test_dataset1[['grade', 'salary']])
plt.scatter(test_grade1, test_salary1, c='orange', label='Test Dataset 1')
print('Test Dataset 1 belongs to Cluster:', test_cluster1)

#Test dataset 2
test_grade2= 3.9
test_salary2 = 800
test_dataset2 = pd.DataFrame({'grade': [test_grade2], 'salary': [test_salary2]})
test_cluster2 = km.predict(test_dataset2[['grade', 'salary']])
plt.scatter(test_grade2, test_salary2, c='purple', label='Test Dataset 2')
print('Test Dataset 2 belongs to Cluster:', test_cluster2)
plt.show()
