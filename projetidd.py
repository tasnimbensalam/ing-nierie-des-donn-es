import numpy as np
import pandas as pd

from pandas import read_csv

#Séparation par colonne

col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=col)

#Partie 1

#Calcul de la moyenne, écart-type, quartiles

res = pd.DataFrame({
    "moy": data[col].mean(),
    "ecartType": data[col].std(),
    "Q1": data[col].quantile(0.25),
    "Q2": data[col].quantile(0.5),
    "Q3": data[col].quantile(0.75)
    })

print(res)

#print(data.describe()) Autre façon de faire

#Calcule des intervalles de confiance à 95% des moyennes

n = data[col].count()
m = data[col].mean()
sd = data[col].std()
se = sd / np.sqrt(n)
z = 1.96

out = pd.DataFrame({
    "n": n,
    "mean": m,
    "ci95_low": m-z*se,
    "ci95_high": m+z*se
    })

print(out)

#Test de la normalité avec le test de Shapiro-Wilk

from scipy.stats import shapiro

shapiro_results = {}

for c in col:
    stat, p = shapiro(data[c])
    shapiro_results[c] = {"statistic": stat, "p_value": p}

shapiro_df = pd.DataFrame(shapiro_results).T
print(shapiro_df)

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

for c in col:
    x = data[c].dropna()

    plt.figure(figsize=(15,4))

    # Histogramme
    plt.subplot(1,3,1)
    plt.hist(x, bins=20)
    plt.title(f"Histogramme de {c}")
    plt.xlabel(c)
    plt.ylabel("Fréquence")

    # Boxplot
    plt.subplot(1,3,2)
    plt.boxplot(x, vert=False)
    plt.title(f"Boxplot de {c}")
    plt.xlabel(c)

    # Courbe KDE
    plt.subplot(1,3,3)
    kde = gaussian_kde(x)
    x_grid = np.linspace(min(x), max(x), 200)
    plt.plot(x_grid, kde(x_grid))
    plt.title(f"Densité (KDE) de {c}")
    plt.xlabel(c)

    plt.tight_layout()
    plt.show()


import seaborn as sns

from scipy.stats import pearsonr
# Analyse bivariée

# 1. Création de la matrice de corrélation (heatmap)
# Nom des colonnes et lecture du fichier
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#  Heatmap : visualiser toutes les corrélations
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr().abs(), annot=True)
plt.show()

# Nombre d'observations
n = data.shape[0]

# 2. Pour chaque paire : calcul de r, R² et IC95 du R²
for x in data.columns:
    for y in data.columns:

        # Coefficient r et p-value
        r, p_value = pearsonr(data[x], data[y])
        #R²
        R2 = r**2

        # Calcul de l'IC95 du R2
        # On évite les erreurs quand r est égal à 1 ou très proche de 1
        if abs(r) >= 0.999999:
            r_inf, r_sup = r, r
            R2_inf, R2_sup = R2, R2
        else:
            Z = 0.5 * np.log((1 + r) / (1 - r))
            s_Z = np.sqrt(1 / (n - 3))
            Z_inf = Z - 1.96 * s_Z
            Z_sup = Z + 1.96 * s_Z
            r_inf = (np.exp(2*Z_inf) - 1) / (np.exp(2*Z_inf) + 1)
            r_sup = (np.exp(2*Z_sup) - 1) / (np.exp(2*Z_sup) + 1)
            R2_inf = r_inf**2
            R2_sup = r_sup**2

        # 3. Tracer la droite de régression uniquement si p < 0.03
        if p_value < 0.03 and x != y:
            plt.figure(figsize=(6, 4))
            sns.regplot(x=data[x], y=data[y])
            plt.title(f"Régression {x} vs {y}")
            plt.show()

        # Affichage de r, R2 et IC95
        print(f"Corrélation {x} vs {y} : r = {r:.2f}, R² = {R2:.2f}, IC95% R² = [{R2_inf:.3f}, {R2_sup:.3f}]")
        print(p_value)

# 4. Identification des 3 variables les plus corrélées avec MEDV
corr_medv = data.corr()['MEDV']

top3_vars = corr_medv.drop('MEDV').abs().sort_values(ascending=False).head(3)

print(top3_vars)

"""1-Calculer la matrice de variance-covariance :
La matrice de variance–covariance sert à décrire la dispersion des variables et leurs relations linéaires.
"""

# 1-Calculer la matrice de variance-covariance
cov_matrix = data.cov()
print("Matrice de variance-covariance :")
print(cov_matrix)

#2. Interprétez les valeurs de variance (diagonale) et de covariance
# Extraire les variances (diagonale)
variances = np.diag(cov_matrix)
print("\nVariances de chaque variable :")
for i, col in enumerate(data.columns):
    print(f"{col}: {variances[i]:.4f}")

"""Interprétation des variances (diagonale)

Les valeurs de variance indiquent la dispersion de chaque variable autour de sa moyenne. On observe par exemple que AGE (792.35), ZN (543.93) et TAX (28 404.76) possèdent des variances très élevées : ce sont des variables très dispersées, avec beaucoup de variation dans le dataset. À l’inverse, des variables comme NOX (0.013), CHAS (0.064) ou RM (0.49) présentent une faible variance, ce qui signifie qu’elles varient peu entre les observations.

Les covariances révèlent des relations importantes : TAX et RAD varient fortement ensemble, tout comme AGE et INDUS. Les relations négatives montrent que les zones résidentielles sont opposées aux zones industrielles et aux zones fortement taxées. Certaines variables, comme CHAS, ont une covariance proche de zéro avec la plupart des autres, indiquant peu de relation linéaire.

3. Identifiez les paires de variables avec les covariances les plus élevées
"""

pairs = []

for i in range(len(cov_matrix.columns)):
    for j in range(i+1, len(cov_matrix.columns)):
        var1 = cov_matrix.columns[i]
        var2 = cov_matrix.columns[j]
        cov_value = cov_matrix.iloc[i, j]
        pairs.append((var1, var2, cov_value))

# Convertir en DataFrame
pairs_df = pd.DataFrame(pairs, columns=["Variable 1", "Variable 2", "Covariance"])

# Trier par covariance
sorted_pairs = pairs_df.sort_values(by="Covariance", ascending=False)

print("\n--- Plus forte covariance positive ---")
print(sorted_pairs.head(10))

"""4. Comparez la matrice de covariance avec la matrice de corrélation : quelles différences observez-vous?"""

# Calculer la matrice de corrélation
corr_matrix = data.corr()

print("\nMatrice de corrélation :")
print(corr_matrix)

print("\nMatrice de covariance :")
print(cov_matrix)

"""La matrice de corrélation est normalisée, chaque valeur étant divisée par l’écart-type des variables correspondantes, ce qui rend les coefficients comparables et indépendants des unités de mesure. En revanche, la matrice de covariance n’est pas normalisée, chaque valeur dépend donc de l’échelle et des unités des variables, ce qui la rend sensible à l’ampleur et à la distribution des données.

5.Calculez les valeurs propres et vecteurs propres de cette matrice
"""

# Calculer les valeurs propres et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print('\nles valeurs propres : ',eigenvalues)
print('\nles vecteurs propres : ',eigenvectors)

"""Analyse en Composantes Principales (ACP)

"""

#1. Standardisez les variables et justifiez cette étape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fitting the Scaler
scaler.fit(data)

# Transforming the Data using the above fitted scaler
scaled_data = scaler.transform(data)
scaled_data
scaled_data = pd.DataFrame(scaled_data, columns = [data.columns])
scaled_data.head()

from sklearn.decomposition import PCA
X = scaled_data.drop(columns=["MEDV"])
# Application de l’ACP
pca = PCA()  # toutes les composantes
X_pca = pca.fit_transform(X)

# Afficher les premières valeurs
X_pca[:5]

"""3. Calculez et visualisez la variance expliquée par chaque composante"""

myPCA = PCA(n_components=13)

# Fitting the data
x = myPCA.fit_transform(X)
print("Variance expliquée par Components :", myPCA.explained_variance_)
print("Percent Variance explained by Components are:", myPCA.explained_variance_ratio_)
print("Cumulative Percent Variance Explained by Components are:",
      np.cumsum(myPCA.explained_variance_ratio_))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# PCA sur toutes les variables
n_components = X.shape[1]
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Variance expliquée
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)


# Scree plot centré sur la variance cumulée
plt.figure(figsize=(10,6))
plt.plot(range(1, n_components+1), explained_variance_ratio, 'o-', label='Variance individuelle')
plt.plot(range(1, n_components+1), cumulative_variance, 's--', label='Variance cumulée')
plt.axhline(y=0.8, color='green', linestyle='--', label='80% variance cumulée')
plt.xlabel('Composantes principales')
plt.ylabel('Variance expliquée')
plt.title('Scree Plot et Variance cumulée')
plt.xticks(range(1, n_components+1))
plt.grid(True)
plt.legend()
plt.show()

"""D’après ce qui précède, nous pouvons constater que 5 composantes suffisent, car elles expliquent 80 % de la variation dans les données.

5. Visualisez le cercle des corrélations : quelles variables contribuent le plus aux PC1 et PC2?
"""

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
cor_var = pca.components_.T * np.sqrt(pca.explained_variance_)
plt.figure(figsize=(6,6))
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)

for i, var in enumerate(X.columns):
    plt.arrow(0, 0, cor_var[i,0], cor_var[i,1],
              color='r', alpha=0.5, head_width=0.03)
    plt.text(cor_var[i,0]*1.15, cor_var[i,1]*1.15, var, color='b', ha='center', va='center')

# Cercle unité
circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='black', linestyle='--')
plt.gca().add_artist(circle)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cercle des corrélations')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.show()

"""6. Interprétez les deux premières composantes principales : que représentent-elles?

![image.png](data:image/png;base64,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)

7. Projetez les observations dans l’espace des 2 premières composantes (biplot)
"""

from sklearn.model_selection import train_test_split

train, test = train_test_split(scaled_data, test_size=0.30, random_state=42)
scores = pca.fit_transform(train)

# Creating a DataFrame
scores_df = pd.DataFrame(scores)
scores_df.columns = ['PCA1', 'PCA2','PCA3', 'PCA4', 'PCA5']
scores_df
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

a4_dims = (12, 9)
fig, ax = plt.subplots(figsize=a4_dims)
ax = myplot(scores[:,0:2], pca.components_.T[:, 0:2],list(scaled_data.columns))
plt.show()

"""8. Identifiez d’éventuels outliers dans cet espace réduit

1 - Sélection de 3 variables fortement corrélées avec MEDV
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

vars_model = ["RM", "LSTAT", "PTRATIO"]

X = data[vars_model]
y = data["MEDV"]

"""2 - Appliquez la régression linéaire multiple"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""3 - Calculez R2, R2 ajusté, RMSE"""

r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² :", r2)
print("R² ajusté :", r2_adj)
print("RMSE :", rmse)

"""4 - Testez la significativité des coefficients (p-values)"""

X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

print(model_sm.summary())

"""5 - Visualisez : valeurs prédites vs valeurs réelles"""

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linewidth=2)

plt.xlabel("Valeurs réelles (MEDV)")
plt.ylabel("Valeurs prédites")
plt.title("Prédictions vs Réel — Régression Multiple")
plt.show()

"""6 - Interprétation de chaque coefficient"""

coeffs = pd.DataFrame({
    "Variable": vars_model,
    "Coefficient": model.coef_
})

print(coeffs)
print("Intercept :", model.intercept_)

# Régression sur composantes principales (PCR)

# Séparer X et y
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Standardisation des variables
# (on met toutes les variables sur la même échelle)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Appliquez une régression linéaire en utilisant les composantes principales comme prédicteurs
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

# 1. Régression linéaire avec les composantes principales
model_pcr = LinearRegression()
model_pcr.fit(X_pca, y)

y_pred_pcr = model_pcr.predict(X_pca)

rmse_pcr = np.sqrt(mean_squared_error(y, y_pred_pcr))
r2_pcr = r2_score(y, y_pred_pcr)

print("Résultats PCR :")
print("RMSE :", rmse_pcr)
print("R2   :", r2_pcr)

# 2. Comparez les performances (RMSE, R2) avec la régression sur variables originales
model_org = LinearRegression()
model_org.fit(X, y)

y_pred_org = model_org.predict(X)

rmse_org = np.sqrt(mean_squared_error(y, y_pred_org))
r2_org = r2_score(y, y_pred_org)

print("\nRégression originale :")
print("RMSE :", rmse_org)
print("R2   :", r2_org)

"""3. Discutez des avantages et inconvénients de cette approche

- Si le RMSE de la PCR est plus petit que celui du modèle original,
  alors la PCR prédit mieux.

- Si le R² de la PCR est plus grand, la PCR explique mieux la variabilité.

- Avantages de la PCR :
  * réduit le nombre de variables
  * enlève les problèmes de corrélation entre variables
  * rend le modèle plus stable

- Inconvénients de la PCR :
  * les composantes ne sont pas interprétables (ce ne sont plus des vraies variables)
  * on peut perdre de l’information si trop de composantes sont supprimées
  * la PCR ne cherche pas à améliorer la prédiction mais seulement la variance

- Conclusion:
  La PCR est utile quand les variables sont nombreuses ou très corrélées.
  Sinon, la régression classique peut être meilleure.

PARTIE 2 : Mall Customers- Clustering
"""

# PARTIE 2 — MALL CUSTOMERS : EXPLORATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("Mall_Customers.csv")

num_vars = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
df = data[num_vars]

# EXPLORATION

# 1. Statistiques descriptives

print("Statistiques descriptives")
print(df.describe())

# 2. Matrice de corrélation

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# 3. Normalisation des données (obligatoire pour ACP)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

"""MATRICE DE VARIANCE - COVARIANCE + ACP"""

# MATRICE DE VARIANCE - COVARIANCE + ACP

# 2. Interprétation

"""
Variances (diagonale) environ égal à 1, les données ont été normalisées.
Covariances proches de 0, les variables sont faiblement corrélées.
Cela suggère que l’ACP peut révéler des axes de variabilité intéressants.
"""

# 1. Matrice de variance-covariance

cov_matrix = np.cov(df_scaled.T)
print("\nMatrice de variance-covariance")
print(cov_matrix)

# 3. Application de l’ACP sur données normalisées

pca = PCA()
pca_result = pca.fit_transform(df_scaled)

variance_expliquee = pca.explained_variance_ratio_
variance_cumulee = np.cumsum(variance_expliquee)

print("\n Variance expliquée par composante")
print(variance_expliquee)
print("\nVariance cumulée")
print(variance_cumulee)

# 4. Détermination du nombre optimal de composantes (scree plot)

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(num_vars) + 1), variance_expliquee, marker="o")
plt.title("Scree Plot")
plt.xlabel("Composante principale")
plt.ylabel("Variance expliquée")
plt.grid()
plt.show()

# 5. Cercle des corrélations

components = pca.components_

plt.figure(figsize=(6, 6))

for i, var in enumerate(num_vars):
    plt.arrow(0, 0, components[0, i], components[1, i],
              color='blue', alpha=0.7)
    plt.text(components[0, i] * 1.1,
             components[1, i] * 1.1,
             var, color='black', fontsize=12)

# Axes du cercle

plt.axhline(0, color='grey')
plt.axvline(0, color='grey')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cercle des corrélations")
plt.grid()
plt.show()

# 6. Projection des clients dans l’espace PC1–PC2

plt.figure(figsize=(7, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection des clients (ACP)")
plt.grid()
plt.show()

# 7. Cette projection permet-elle d’identifier visuellement des groupes ?

"""
OUI : Visuellement, plusieurs blocs apparaissent.

On distingue notamment :
- Un groupe "haut revenu" vers la droite (PC1+)
- Un groupe "fort spending score" vers le haut (PC2+)
- Un groupe "faible revenu + faible score" en bas à gauche
- Un groupe plus âgé et modéré sur Spending Score (zone centrale)

Conclusion : l’ACP révèle des structures naturelles,
ce qui confirme que du clustering (K-Means) est pertinent.
"""

"""#K-means

"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("QUESTION 1 : Détermination de k optimal")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Méthode du coude
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, km.labels_))

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbe du coude
axes[0].plot(range(2, 11), wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel("Nombre de clusters (k)")
axes[0].set_ylabel("WCSS (Inertie)")
axes[0].set_title("Méthode du Coude")
axes[0].set_xticks(range(2, 11))
axes[0].grid(True, alpha=0.3)

# Score Silhouette
axes[1].plot(range(2, 11), silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel("Nombre de clusters (k)")
axes[1].set_ylabel("Score Silhouette")
axes[1].set_title("Score Silhouette par k")
axes[1].set_xticks(range(2, 11))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.show()

print(f"\nScores Silhouette: {dict(zip(K_range, [round(s, 3) for s in silhouette_scores]))}")
print(f"k optimal suggéré par silhouette: k = {list(K_range)[np.argmax(silhouette_scores)]}")

#k optimal =5
#2.. Appliquez K-Means avec k optimal sur les données originales
#5 clusters
km1=KMeans(n_clusters=5)
#fitting the input data
km1.fit(X)
#predecting the lables of the input data
y=km1.predict(X)
#adding the labels to a column named label
data["label"]=y
data.head()

#3. Appliquez K-Means sur les composantes principales retenues
#PCA → garder 2 ou 3 composantes (celles qui expliquent 80–95% de la variance)
from sklearn.decomposition import PCA
# Extraction des 3 composantes principales
X_pca = pca_result[:, :3]  # Garde PC1, PC2, PC3

# Application de K-Means avec k=5 (déterminé par la méthode du coude)
km_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_pca = km_pca.fit_predict(X_pca)

# Ajout au dataframe
data['Cluster_PCA'] = labels_pca

#4. Visualisez en scatter plot : Income vs Spending Score (coloré par cluster)
#Scatterplot of the clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label",
                 palette=['green','orange','brown','dodgerblue','red'], legend='full',data = data  ,s = 60 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()

#Caractériser
# K-Means cluster characteristics
# Caractériser les clusters
for cluster in sorted(data['label'].unique()):
    print(f"\nCluster {cluster}:")
    print(data[data['label'] == cluster][["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())

# QUESTION 5 : Caractérisation de chaque cluster

# --- 5.1 Moyennes par cluster ---
print("\n--- MOYENNES PAR CLUSTER (Données originales) ---")
cluster_means = data.groupby("label")[num_vars].mean()
print(cluster_means.round(2))

# --- 5.2 Taille de chaque cluster ---
print("\n--- TAILLE DES CLUSTERS ---")
cluster_sizes = data["label"].value_counts().sort_index()
print(cluster_sizes)

# --- 5.3 Répartition par genre ---
print("\n--- RÉPARTITION PAR GENRE ---")
gender_by_cluster = pd.crosstab(data["label"], data["Genre"], normalize='index') * 100
print(gender_by_cluster.round(1))

# --- 5.4 Profil type de chaque cluster ---
print("\n--- PROFIL TYPE PAR CLUSTER ---")

profiles = []
for cluster in range(5):
    cluster_data = data[data["label"] == cluster]

    avg_age = cluster_data["Age"].mean()
    avg_income = cluster_data["Annual Income (k$)"].mean()
    avg_spending = cluster_data["Spending Score (1-100)"].mean()
    size = len(cluster_data)
    pct_female = (cluster_data["Genre"] == "Female").mean() * 100

    # Définition du profil
    if avg_income > 70 and avg_spending > 60:
        profile = " High Income, High Spenders (VIP)"
    elif avg_income > 70 and avg_spending < 40:
        profile = " High Income, Low Spenders (Économes)"
    elif avg_income < 50 and avg_spending > 60:
        profile = " Low Income, High Spenders (Impulsifs)"
    elif avg_income < 50 and avg_spending < 40:
        profile = " Low Income, Low Spenders (Prudents)"
    else:
        profile = " Average (Modérés)"

    profiles.append({
        "Cluster": cluster,
        "Taille": size,
        "Age_moy": round(avg_age, 1),
        "Income_moy": round(avg_income, 1),
        "Spending_moy": round(avg_spending, 1),
        "% Femmes": round(pct_female, 1),
        "Profil": profile
    })

    print(f"\nCluster {cluster} - {profile}")
    print(f"  • Taille: {size} clients")
    print(f"  • Âge moyen: {avg_age:.1f} ans")
    print(f"  • Revenu moyen: {avg_income:.1f} k$")
    print(f"  • Score dépenses: {avg_spending:.1f}")
    print(f"  • % Femmes: {pct_female:.1f}%")

# Tableau récapitulatif
profiles_df = pd.DataFrame(profiles)
print("\n--- TABLEAU RÉCAPITULATIF ---")
print(profiles_df.to_string(index=False))

# Visualisation des caractéristiques des clusters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot Age par cluster
data.boxplot(column="Age", by="label", ax=axes[0, 0])
axes[0, 0].set_title("Âge par Cluster")
axes[0, 0].set_xlabel("Cluster")
axes[0, 0].set_ylabel("Âge")

# Boxplot Income par cluster
data.boxplot(column="Annual Income (k$)", by="label", ax=axes[0, 1])
axes[0, 1].set_title("Revenu par Cluster")
axes[0, 1].set_xlabel("Cluster")
axes[0, 1].set_ylabel("Revenu (k$)")

# Boxplot Spending Score par cluster
data.boxplot(column="Spending Score (1-100)", by="label", ax=axes[1, 0])
axes[1, 0].set_title("Score Dépenses par Cluster")
axes[1, 0].set_xlabel("Cluster")
axes[1, 0].set_ylabel("Spending Score")

# Répartition par genre
gender_by_cluster.plot(kind='bar', ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
axes[1, 1].set_title("Répartition par Genre")
axes[1, 1].set_xlabel("Cluster")
axes[1, 1].set_ylabel("Pourcentage (%)")
axes[1, 1].legend(title="Genre")
axes[1, 1].tick_params(axis='x', rotation=0)

plt.suptitle("", y=1.02)
plt.tight_layout()
plt.show()

#6. Comparez les résultats K-Means sur données originales vs sur composantes principales
print("Comparaison Original vs PCA")

# Comparaison des tailles de clusters
print("\n--- TAILLES DES CLUSTERS ---")
comparison = pd.DataFrame({
    "KMeans_Original": data["label"].value_counts().sort_index(),
    "Cluster_PCA": data["Cluster_PCA"].value_counts().sort_index()
})
print(comparison)

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

colors = ['green','orange','brown','dodgerblue','red']

# Comparaison dans l'espace PC1-PC2
for cluster in range(5):
    mask = data["label"] == cluster
    axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1],
                    c=colors[cluster], label=f"Cluster {cluster}", alpha=0.7, s=50)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("Clusters KMeans projetés sur PCA")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for cluster in range(5):
    mask = data["Cluster_PCA"] == cluster
    axes[1].scatter(pca_result[mask, 0], pca_result[mask, 1],
                    c=colors[cluster], label=f"Cluster {cluster}", alpha=0.7, s=50)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Clusters PCA dans l'espace PCA")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

"""**Classification Hiérarchique Ascendante (CAH)**"""

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Q1. Construisez des dendrogrammes avec : complete, average, Ward

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

methods = ["complete", "average", "ward"]
for m in methods:
    plt.figure(figsize=(8,5))
    Z = linkage(X, method=m)
    dendrogram(Z)
    plt.title(f"Dendrogramme - Méthode {m.capitalize()}")
    plt.xlabel("Clients")
    plt.ylabel("Distance")
    plt.show()



# Q2. Choisissez le meilleur critère et coupez le dendrogramme

# On choisit Ward pour homogénéité et on coupe en 5 clusters
Z_ward = linkage(X, method='ward')
clusters = fcluster(Z_ward, 5, criterion='maxclust')
data['CAH_label'] = clusters

print("\n--- Extrait des clients avec leurs clusters CAH ---")
print(data[['Annual Income (k$)', 'Spending Score (1-100)', 'CAH_label']].head())



# Q3. Matrice de confusion : comparez K-Means vs CAH

# K-Means avec 5 clusters
k_opt = 5
km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
data['label'] = km.fit_predict(X)
data['Cluster_Original'] = data['label']  # pour cohérence

# Matrice de correspondance
correspondence = pd.crosstab(data['Cluster_Original'], data['CAH_label'],
                             rownames=['KMeans'], colnames=['CAH'])
print("\n--- Matrice de correspondance K-Means vs CAH ---")
print(correspondence)

# Q4. Les deux méthodes identifient-elles les mêmes groupes ?

# Réponse simple : Les groupes ne sont pas exactement les mêmes,
# mais on voit qu'il y a beaucoup de similarités entre K-Means et CAH.
# Certains clients changent de cluster selon la méthode,
# mais globalement, les grandes tendances se ressemblent.


# Q5. Appliquez la CAH sur les composantes principales : observe-t-on une meilleure séparation ?

# Normalisation pour PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ACP → garder 2 composantes
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# CAH sur les deux premières composantes
Z_pca = linkage(pca_result, method='ward')

# Dendrogramme
plt.figure(figsize=(8,5))
dendrogram(Z_pca)
plt.title("Dendrogramme CAH sur 2 premières composantes (Ward)")
plt.xlabel("Clients")
plt.ylabel("Distance")
plt.show()

# Couper le dendrogramme en 5 clusters
clusters_pca = fcluster(Z_pca, 5, criterion='maxclust')
data['CAH_PCA_label'] = clusters_pca

# Scatter plot des clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=data['CAH_PCA_label'],
                palette='Set1', s=60)
plt.title("CAH sur 2 premières composantes principales")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster CAH")
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- Extrait des clients avec leurs clusters PCA ---")
print(data[['Annual Income (k$)', 'Spending Score (1-100)', 'CAH_PCA_label']].head())

"""**Evaluation**"""

#Évaluation et validation 1. Calculez le coefficient de silhouette pour chaque méthode
# Scores de qualité
silhouette_original = silhouette_score(X, data['Cluster_Original'])
silhouette_pca = silhouette_score(pca_result, data['CAH_PCA_label'])

print(f"Score Silhouette (Original / K-Means):  {silhouette_original:.4f}")
print(f"Score Silhouette (PCA / CAH):           {silhouette_pca:.4f}")

# EVALUATION DES MODELES

from sklearn.metrics import silhouette_score, davies_bouldin_score


X_original = X
X_scaled_model = X_scaled
X_pca_model = X_pca

labels_kmeans = data["Cluster_Original"]
labels_kmeans_pca = data["Cluster_PCA"]
labels_cah = data["CAH_label"]
labels_cah_pca = data["CAH_PCA_label"]

sil_kmeans = silhouette_score(X_original, labels_kmeans)
sil_kmeans_pca = silhouette_score(X_pca_model, labels_kmeans_pca)
sil_cah = silhouette_score(X_original, labels_cah)
sil_cah_pca = silhouette_score(X_pca_model, labels_cah_pca)

print(f"Silhouette - KMeans (original) :   {sil_kmeans:.4f}")
print(f"Silhouette - KMeans (PCA)      :   {sil_kmeans_pca:.4f}")
print(f"Silhouette - CAH (Ward)        :   {sil_cah:.4f}")
print(f"Silhouette - CAH (PCA)         :   {sil_cah_pca:.4f}")

db_kmeans = davies_bouldin_score(X_original, labels_kmeans)
db_kmeans_pca = davies_bouldin_score(X_pca_model, labels_kmeans_pca)
db_cah = davies_bouldin_score(X_original, labels_cah)
db_cah_pca = davies_bouldin_score(X_pca_model, labels_cah_pca)

print(f"Davies-Bouldin - KMeans (original) : {db_kmeans:.4f}")
print(f"Davies-Bouldin - KMeans (PCA)      : {db_kmeans_pca:.4f}")
print(f"Davies-Bouldin - CAH (Ward)        : {db_cah:.4f}")
print(f"Davies-Bouldin - CAH (PCA)         : {db_cah_pca:.4f}")

"""On remarque que le coefficient le plus élevé de silhouette pour chaque méthode est avec la méthode KMeans originale et l'indice le plus faible est obtenu avec la méthode KMeans originale malgré que la méthode de CAH (WARD) donne aussi de très bonnes valeurs. La meilleure méthode reste le KMeans original, ce qui nous montre aussi que l'ACP n'améliore pas la qualité du clustering dans notre cas."""