
Ce dépôt contient l'ensemble des travaux pratiques et le projet final réalisés dans le cadre du module **Ingénierie des Données**.



**Cursus :** Master 1 Informatique  

**Université :** Université Sorbonne Paris Nord  

**Année :** 2025-2026



##  Contexte du Projet



L'objectif de ce module était de maîtriser la chaîne complète de traitement de la donnée (ETL : Extract, Transform, Load), depuis la récupération de données brutes sur le web jusqu'à leur stockage et leur analyse.



Le projet final se concentre sur la constitution d'une base de données de véhicules d'occasion (fichier `voitures.db`) afin d'analyser les tendances du marché.



##  Technologies Utilisées



* **Langage :** Python 3

* **Collecte de données (Scraping) :** BeautifulSoup / Requests / Selenium

* **Manipulation de données :** Pandas, NumPy

* **Base de données :** SQLite (SQLAlchemy)

* **Visualisation :** Matplotlib, Seaborn

* **Environnement :** Jupyter Notebooks



##  Structure du Dépôt



### 1. Le Projet Final (`ProjetIDD.ipynb`, `projetidd.py`)

Le cœur du projet qui implémente un pipeline de données complet :

* **Extraction :** Scraping automatisé d'annonces automobiles (marque, modèle, prix, année, kilométrage).

* **Nettoyage :** Traitement des valeurs manquantes, conversion des types et normalisation des données.

* **Stockage :** Création et alimentation d'une base de données relationnelle SQLite (`voitures.db`).

* **Analyse :** Exploration des données pour comprendre les corrélations (ex: évolution du prix selon le kilométrage).



### 2. Travaux Pratiques (TP2 à TP5)

Exercices progressifs ayant mené à la réalisation du projet :

* **TP2 & TP3 :** Prise en main du Web Scraping et extraction de données structurées.

* **TP4 :** Modélisation de base de données et requêtes SQL via Python.

* **TP5 :** Analyse exploratoire et visualisation de données (Data Viz).



##  Comment lancer le projet



1.  Cloner le dépôt :

    ```bash

    git clone [https://github.com/votre-username/data-engineering-cars-project.git](https://github.com/votre-username/data-engineering-cars-project.git)

    ```

2.  Installer les dépendances requises :

    ```bash

    pip install pandas requests beautifulsoup4 matplotlib sqlalchemy

    ```

3.  Ouvrir le notebook principal :

    ```bash

    jupyter notebook ProjetIDD.ipynb

    ```



---

*Ce projet a été réalisé dans un but pédagogique.*
