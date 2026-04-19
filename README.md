# Prédiction du Prix des Logements — Mini Projet Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/Licence-MIT-green)
![Status](https://img.shields.io/badge/Statut-En%20cours-yellow)

> Mini Projet Machine Learning — Filière 1ère MR-BC | AU 2025/2026  
> Responsable : Dr. Fadoua BOUAFIF

---

## Équipe

| Étudiant | GitHub |
|---|---|
| *YANN HAROLD TCHIOFFO FOTSO* | [@username](https://github.com/Ya-rold) |


---

##  Description du projet

Ce projet s'inscrit dans le cadre du mini-projet de Machine Learning de la filière 1ère MR-BC. L'objectif est d'appliquer plusieurs approches de machine learning pour résoudre un problème métier concret.

**Problématique :** Nous cherchons à prédire automatiquement le **prix médian de logements** en Californie à partir de leurs caractéristiques (localisation, surface, revenu du quartier, etc.), en comparant les performances de plusieurs algorithmes supervisés.

**Dataset :** [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) — 20 640 observations, 10 variables.

---

## 🗂️ Structure du dépôt

```
📦 mini-projet-ml/
├── 📂 data/
│   └── housing.csv                  # Dataset California Housing
├── 📂 notebooks/
│   ├── 01_exploration.ipynb         # Analyse exploratoire des données (EDA)
│   ├── 02_preprocessing.ipynb       # Nettoyage et préparation des données
│   ├── 03_modelisation.ipynb        # Entraînement et comparaison des modèles
│   └── 04_resultats.ipynb           # Visualisations et interprétation des résultats
├── 📂 src/
│   ├── preprocessing.py             # Fonctions de prétraitement
│   ├── models.py                    # Définition et entraînement des modèles
│   └── evaluation.py                # Métriques et visualisations
├── 📂 figures/
│   ├── distribution_prix.png        # Distribution de la variable cible
│   ├── comparaison_modeles.png      # Graphique comparatif des métriques
│   └── importance_variables.png     # Importance des variables (Random Forest)
├── 📂 rapport/
│   └── Rapport_Mini_Projet_ML_2026.md  # Rapport complet du projet
├── requirements.txt                 # Dépendances Python
└── README.md                        # Ce fichier
```

---

##  Méthodes comparées

| # | Algorithme | Famille | Librairie |
|---|---|---|---|
| 1 | Régression Linéaire | Méthodes linéaires | `scikit-learn` |
| 2 | Random Forest | Ensemble — Bagging | `scikit-learn` |
| 3 | XGBoost | Ensemble — Boosting | `xgboost` |
| 4 | MLP (Réseau de neurones) | Deep Learning | `scikit-learn` |

---

##  Résultats

> *Tableau à compléter après exécution complète des modèles.*

| Modèle | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|
| Régression Linéaire | ~0.73 | ~0.53 | ~0.60 |
| Random Forest | ~0.51 | ~0.33 | ~0.80 |
| XGBoost | **~0.45** | **~0.30** | **~0.84** |
| MLP | ~0.55 | ~0.37 | ~0.78 |

---

## Installation et utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/mini-projet-ml.git
cd mini-projet-ml
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer les notebooks

```bash
jupyter notebook notebooks/
```

Ou ouvrir directement sur **Google Colab** :

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Dépendances (`requirements.txt`)

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.4
xgboost>=2.0
jupyter>=1.0
```


---

##  Références

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
- Pace, R.K., & Barry, R. (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters.
- [Kaggle — California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu)

---

##  Licence

Ce projet est réalisé dans un cadre académique — Département Informatique & Méthodes Quantitatives, AU 2025/2026.
