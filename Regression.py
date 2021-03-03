#!/usr/bin/env python
# coding: utf-8

# # Exercice 2 : Popularity prediction and data analysis

# In[191]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import mean_squared_error, make_scorer, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold 

from sklearn.ensemble import IsolationForest, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA
from sklearn.manifold import t_sne

from sklearn.pipeline import make_pipeline

from scipy.stats import ttest_ind

import warnings


# In[141]:


warnings.filterwarnings('ignore')


# In[182]:


data = pd.read_csv('Spotify_exo2.csv')
y = data['popularity']
X = data.drop(['popularity', 'genres'], axis = 1)


# # Dataset analysis

# ## Global info

# In[7]:


data.shape


# In[8]:


data.head(15)


# In[58]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[6]:


data.dtypes.value_counts()


# ## Distribution of quantitative features

# In[10]:


plt.figure(figsize = (27,22))
for index, col in zip(range(1, 14), data.drop('genres', axis = 1).columns):
    plt.subplot(4, 4, index)
    sns.distplot(data[col])


# ## Genre feature

# In[24]:


data['genres'].unique().shape


# ## Visualisation

# In[26]:


sns.pairplot(data)


# ## Outliers

# In[5]:


df = data.drop('genres', axis = 1)
outlier_detector = IsolationForest(contamination = 0.01)
outlier_detector.fit(df)
data[outlier_detector.predict(df) == -1]


# ## Correlation

# In[13]:


plt.figure(figsize = (12, 8))
sns.heatmap(data.corr(), cmap = 'Blues')
data.corr()


# # Preprocessing and Feature selection

# ## Train set and test set

# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[144]:


X_train.shape


# ## Normalisation

# In[145]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_c = np.array(X_train)
y_train = y_train.to_numpy()
y_train = scaler.fit_transform(y_train.reshape((y_train.shape[0], 1)))


# ## Idea - Implementation - Evaluation cycle

# In[146]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_score = make_scorer(rmse, greater_is_better = False)


# In[147]:


def evaluation(model_name, model):
    rmse = cross_val_score(model, X_train, y_train, cv = 5, scoring = rmse_score)
    r2 = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'r2')
    print('RMSE with cross_val : ', " Mean  ", np.mean(rmse), "\t Std  ", np.std(rmse))
    print('\n R2 with cross_val : ', " Mean ", np.mean(r2), "\t Std : ", np.std(r2))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 5, 
                                               scoring = rmse_score,
                                               train_sizes = np.linspace(0.1, 1, 10), random_state = 0)
    plt.figure(figsize = (12, 8))
    plt.plot(N, train_score.mean(axis=1), label = 'Train score')
    plt.plot(N, val_score.mean(axis=1), label = 'Val score')
    plt.xlabel('Number of data')
    plt.ylabel('RMSE')
    plt.title(f'Learning curve for {model_name}')
    plt.legend()


# ### Basic model to test ideas

# In[21]:


first_model = LinearRegression()
evaluation('Linear Regression', first_model)


# ### Observations :
# 
# - Underfitting 
# - Let's try to add features

# In[137]:


# Try a polynomial extension
X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 75)
X_train = selector.fit_transform(X_train, y_train)


# In[138]:


evaluation('Degree 2 Polynomial Regression', first_model)


# In[86]:


# Degree 5
X_train = X_train_c
poly = PolynomialFeatures(degree = 5)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 6)
X_train = selector.fit_transform(X_train, y_train)


# In[87]:


evaluation('Degree 5 Polynomial Regression', first_model)


# In[89]:


# Degree 4
X_train = X_train_c
poly = PolynomialFeatures(degree = 4)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 21)
X_train = selector.fit_transform(X_train, y_train)


# In[90]:


evaluation('Degree 4 Polynomial Regression', first_model)


# In[101]:


# Degree 3 (attempt with different numbers of features, here for k = 100)
X_train = X_train_c
poly = PolynomialFeatures(degree = 3)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 108)
X_train = selector.fit_transform(X_train, y_train)


# In[102]:


evaluation('Degree 3 Polynomial Regression', first_model)


# In[103]:


# Back to degree 2 with all features
X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)


# In[104]:


evaluation('Degree 2 Polynomial Regression', first_model)


# # Model selection

# ## Evaluate different models

# In[90]:


X_train = X_train_c
Lasso = Lasso(random_state = 0)
Ridge = Ridge(random_state = 0)
ElasticNet = ElasticNet(random_state = 0)
SVM = SVR()
GPR  = GaussianProcessRegressor(random_state = 0)
RandomForest = RandomForestRegressor(random_state = 0)
AdaBoost = AdaBoostRegressor(random_state = 0)
GradientBoosting = GradientBoostingRegressor(random_state = 0)
MLP = MLPRegressor(random_state = 0)
models = {'Lasso' : Lasso, 'Ridge' : Ridge, 'Elastic Net' : ElasticNet,
          'SVM' : SVM, 'GPR' : GPR,  'RandomForest' : RandomForest, 'AdaBoost' : AdaBoost, 
          'GradientBoosting' : GradientBoosting, 'MLP' : MLP} 


# ### With the basic variables

# In[198]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# ### With a polynomial extension degree 2 with 50 features

# In[199]:


X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 50)
X_train = selector.fit_transform(X_train, y_train)


# In[201]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# ### Polynomial extension 3 with 80 features

# In[203]:


X_train = X_train_c
poly = PolynomialFeatures(degree = 3)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 80)
X_train = selector.fit_transform(X_train, y_train)


# In[204]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# ### By keeping only the 8 best  features

# In[205]:


X_train = X_train_c
selector = SelectKBest(f_classif, k = 8)
X_train = selector.fit_transform(X_train, y_train)


# In[207]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# ### Observations 
# 
# - Most promising models : SVM, RandomForest, GradientBoosting, MLP
# - Polynomial extension doesn't provide better results (overfitting basically)

# # Model Optimisation

# ## SVM

# In[139]:


X_train = X_train_c 
SVM = SVR()


# In[224]:


params = {'C' : [0.1, 10, 100, 1000, 10000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
svm_grid = GridSearchCV(SVM, params, scoring = rmse_score, cv = 5)
svm_grid.fit(X_train, y_train)


# In[226]:


print(svm_grid.best_estimator_)
print(svm_grid.best_score_)


# In[228]:


svm_params = {'C' : [10, 20, 30, 40, 50], 'gamma' : [0, 0.1, 0.5, 1], 'kernel' : ['rbf', 'linear', 'poly', 'sigmoid']}
optimized_svm = RandomizedSearchCV(SVM, svm_params, scoring = rmse_score, cv = 5, n_iter = 10)
optimized_svm.fit(X_train, y_train)


# In[229]:


print(optimized_svm.best_estimator_)
print(optimized_svm.best_score_)


# In[62]:


SVM = make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k = 50), SVR())
params = {'selectkbest__k' : [30, 60, 90], 
          'svr__C' : [10, 15, 20, 25], 
          }
svm_grid = GridSearchCV(SVM, params, scoring = rmse_score, cv = 5)
svm_grid.fit(X_train, y_train)


# In[63]:


print(svm_grid.best_estimator_)
print(svm_grid.best_score_)


# In[44]:


SVM = SVR(C = 4)
evaluation('SVM', SVM)


# ## Random Forest

# In[71]:


X_train = X_train_c
tree_params = {'n_estimators' : [100, 500, 1000, 1500, 2000], 'max_depth': [None, 5, 10, 15, 20]}
tree_grid = GridSearchCV(RandomForestRegressor(random_state = 0), tree_params, scoring = rmse_score, cv = 5)
tree_grid.fit(X_train, y_train)


# In[72]:


print(tree_grid.best_estimator_)
print(tree_grid.best_score_)


# In[78]:


evaluation('Random Forest', tree_grid.best_estimator_)


# ## Gradient Boosting

# In[150]:


GradientBoosting = make_pipeline(PolynomialFeatures(3), SelectKBest(k = 50), GradientBoostingRegressor(random_state = 0))
gb_params = {'gradientboostingregressor__n_estimators' : [100, 1000, 2000], 
             'gradientboostingregressor__max_depth' : [1, 10, 20, 30]}
gb_grid = GridSearchCV(GradientBoosting, gb_params, scoring = rmse_score, cv = 5)
gb_grid.fit(X_train, y_train)


# In[151]:


print(gb_grid.best_estimator_)
print(gb_grid.best_score_)


# In[85]:


evaluation('Gradient Boosting', gb_grid.best_estimator_)


# ## MLP

# In[17]:


X_train = X_train_c

params = {
    'hidden_layer_sizes': [(10,)*i for i in range(10)]+[(40,)*i for i in range(10)]+[(100,)*i for i in range(10)], 
    'activation': ['logistic', 'relu'], 
    'alpha': [10**-i for i in range(10)]
    }

grid = GridSearchCV(MLPRegressor(random_state=0), params, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

grid.fit(X_train, y_train)


# In[18]:


print(grid.best_params_)
print(grid.best_score_)


# ## Ensemble method (stacking)

# In[21]:


X_train = X_train_c
MLP = MLPRegressor(activation = 'relu', alpha = 1, hidden_layer_sizes = (100))

estimators = [
    ('RandomForest', RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)),
    ('MLP', MLP)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=MLP)


scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(scores)
print(scores.mean())
print(scores.std())


# In[45]:


X_train = X_train_c
MLP = MLPRegressor(activation = 'relu', alpha = 1, hidden_layer_sizes = (100), random_state=0)

estimators = [
    ('RandomForest', RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)),
    ('MLP', MLP),
    ('SVM', SVR(C = 4)),
    ('Gradient Boosting', GradientBoostingRegressor(max_depth=10, n_estimators=2000))
]

stacking_model_2 = StackingRegressor(estimators=estimators, final_estimator=MLP)


scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(scores)
print(scores.mean())
print(scores.std())


# ## Evaluate the final model on the test set

# In[209]:


X_train = X_train_c
MLP = MLPRegressor(activation = 'relu', alpha = 1, hidden_layer_sizes = (100), random_state=0)

estimators = [
    ('RandomForest', RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)),
    ('MLP', MLP),
    ('SVM', SVR(C = 4)),
    ('Gradient Boosting', GradientBoostingRegressor(max_depth=10, n_estimators=2000))
]

final_model = StackingRegressor(estimators=estimators, final_estimator=MLP)
final_model.fit(X_train, y_train)


# In[212]:


scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test.reshape((y_test.shape[0], 1)))


# In[217]:


y_pred = final_model.predict(X_test)
print('RMSE : ', rmse(y_test, y_pred))
print('R2 :', r2_score(y_test, y_pred))


# # Handling the feature 'genre'

# ## Method 1 : create new genres with clustering methods

# ### Compare KMeans, hierarchical clustering and Gaussian Mixture

# In[47]:


# will contain scores for each value of K and for each model : KMeans (col 0), GaussianMixture(col 1), 
# and CAH (col 2)
K_silhouette_scores = np.zeros((16, 3))
inertia = []
M = hierarchy.linkage(X_train, method = 'ward', metric = 'euclidean')
for K in range(5,21):
    # Testing different models
    clu_models = {'KMeans': KMeans(n_clusters = K, random_state=0),
              'Gaussian': GaussianMixture(n_components = K, covariance_type = 'full', random_state=0), 
              'Hierarchical' : M}
    index = 0
    for model_name, model in clu_models.items():
        if(model_name == 'Hierarchical'):
            clusters = hierarchy.fcluster(model, t = K, criterion = 'maxclust')
        else:
            clusters = model.fit_predict(X_train)
        # Computation of the silhouette score for each model
        K_silhouette_scores[K-5, index] = silhouette_score(X_train, clusters)
        # For KMeans, we compute the inertia for each value of K (to apply next the elbow method)
        if model_name == 'KMeans':
           inertia.append(model.inertia_)
        index += 1  


# In[48]:


print("Score for different values of K and for the different models \n", K_silhouette_scores)
print("\n Mean score for each model \n",K_silhouette_scores.mean(axis=0))
print("\n Mean score for each value K \n", K_silhouette_scores.mean(axis=1))
# Evolution of the Silhouette score depending on K
plt.figure(figsize=(15,10))
plt.plot(range(5,21),  K_silhouette_scores.mean(axis=1))
plt.xlabel('K number of clusters')
plt.ylabel('Silhouette score')
# Elbow method to determine the number of clusters to keep
plt.figure(figsize=(15,10))
plt.plot(range(5, 21), inertia)
plt.xlabel('K number of clusters')
plt.ylabel('Inter-cluster inertia')
plt.show()
#Dendrogram
plt.figure(figsize=(15,10))
hierarchy.dendrogram(M)
plt.show()


# In[106]:


### K = 10 clusters with KMeans
KMeans = KMeans(n_clusters = 10)
clusters = KMeans.fit_predict(X_train)
clusters = clusters.reshape(clusters.shape[0], 1)


# In[100]:


X_train = np.concatenate((X_train, clusters), axis = 1)


# In[102]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# ## Method 2 : predict genres with a genre classifier 

# In[132]:


data_genre = pd.read_csv('Spotify_train_dataset.csv')
data_genre = data_genre.drop('time_signature', axis = 1)


# In[133]:


LogRegression = LogisticRegression(penalty ='l2', C = 2000, max_iter = 10000, n_jobs = -1)
log_regression = make_pipeline(PolynomialFeatures(2), SelectKBest(k = 70), LogRegression)
MLP = MLPClassifier(activation = 'logistic', alpha = 1e-06, hidden_layer_sizes = (100, 100, 100))

estimators = [
    ('RandomForest', RandomForestClassifier(n_estimators = 1500,  max_depth = 14, random_state = 0, n_jobs = -1)),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('LogisticRegression', log_regression),
    ('GNB', GaussianNB()),
    ('MLP', MLP)
]

genre_classifier = StackingClassifier(estimators = estimators, final_estimator = MLP)


# In[134]:


X_genre_train = data_genre.drop(data_genre.select_dtypes('object').columns, axis = 1)
y_genre_train = data_genre['genre']


# In[135]:


encoder = LabelEncoder()
y_genre_train = encoder.fit_transform(y_genre_train)


# In[136]:


scaler = StandardScaler()
X_genre_train = scaler.fit_transform(X_genre_train)


# In[137]:


genre_classifier.fit(X_genre_train, y_genre_train)


# In[142]:


X_train = X_train_c
genres = genre_classifier.predict(X_train)
genres = genres.reshape(genres.shape[0], 1)


# In[143]:


X_train = np.concatenate((X_train, genres), axis = 1)


# In[145]:


for name, model in models.items():
    print("\n", name, "\n")
    evaluation(name, model)


# # Patterns, visualisation, interpretation...

# ### Transform 'popularity' into a categorical feature

# In[152]:


data.describe()['popularity']


# In[183]:


def categorize_pop(x):
    if x < data.describe()['popularity'][4]:
        return 0
    elif x > data.describe()['popularity'][4] and x < data.describe()['popularity'][5]:
        return 1
    elif x > data.describe()['popularity'][5] and x < data.describe()['popularity'][6]:
        return 2
    else:
        return 3


# In[184]:


data['popularity'] = data['popularity'].map(categorize_pop)


# In[155]:


data


# In[167]:


popularity_1 = data[data['popularity'] == 0]
popularity_2 = data[data['popularity'] == 1]
popularity_3 = data[data['popularity'] == 2]
popularity_4 = data[data['popularity'] == 3]


# ## Statistics for the different categories of popularity

# In[162]:


popularity = ['unpopular', 'not very popular', 'popular', 'very popular']
for i, name in zip(range(4), popularity) :
    print("\n", name, "\n", data[data['popularity'] == i].describe())


# ### Distribution of features by category

# In[192]:


plt.figure(figsize = (32, 25))
for index, col in zip(range(1, 13), data.drop(['genres', 'popularity'], axis = 1)):
    plt.subplot(4, 3, index)
    sns.distplot(popularity_1[col], label = 'unpopular')
    sns.distplot(popularity_2[col], label = 'not very popular')
    sns.distplot(popularity_3[col], label = 'popular')
    sns.distplot(popularity_4[col], label = 'very popular')
    plt.legend()


# ### Boxplots

# In[219]:


plt.figure(figsize = (20, 14))
for index, col in zip(range(1, 13), data.drop(['genres', 'popularity'], axis = 1).columns):
    plt.subplot(3, 4, index)
    sns.boxplot(x = 'popularity', y = col, data = data)


# In[223]:


sns.pairplot(data, hue = 'popularity', palette = sns.color_palette('hls', 4))


# ## Statistical hypothesis testings

# In[174]:


def test(alpha, col):
    stat, p = ttest_ind(popularity_1[col], popularity_4[col])
    if p < alpha : 
        print('H0 rejetée')
    else :
        print('On ne peut pas rejeter H0')


# In[178]:


test(0.05, 'acousticness')
test(0.05, 'energy')


# ## Dimension reduction

# ## PCA

# In[192]:


PCA = PCA(random_state = 0)
data = PCA.fit_transform(data.drop('genres', axis = 1))


# In[266]:


print(np.cumsum(PCA.explained_variance_ratio_))


# In[204]:


plt.figure(figsize = (12,8))
plt.scatter(data[:, 0], data[:, 1], c = data[:, -2])
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')


# In[205]:


plt.figure(figsize = (12,8))
plt.scatter(data[:, 2], data[:, 3], c = data[:, -2])
plt.xlabel('Composante principale 3')
plt.ylabel('Composante principale 4')


# In[206]:


plt.figure(figsize = (12, 8))
plt.scatter(data[:, 4], data[:, 5], c = data[:, -2])
plt.xlabel('Composante principale 5')
plt.ylabel('Composante principale 6')


# In[ ]:




