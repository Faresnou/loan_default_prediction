# Loan Default Prediction

Prédiction automatique du défaut de remboursement d’un prêt immobilier (MORTGAGE)

## Application déployée

**Streamlit :** <https://loans-default-predictions.streamlit.app>

## Description du projet

Ce projet implémente un pipeline ML complet pour la prédiction du risque de défaut de remboursement sur le dataset Loan Default Prediction (Kaggle). Le dataset contient 148 670 dossiers de prêts immobiliers avec des différentes variables relatives aux dossiers des clients demandeurs de crédits.

**Source dataset :** Kaggle — Loan Default Prediction 
**Taille :** 148 670 lignes, 34 features  
**Tâche ML :** Classification binaire 
**Variable cible :** `Status` — 0 (remboursement normal) / 1 (defaut de paiement)  
**Défi principal :** Déséquilibre modéré des classes + détection et correction d’un data leakage temporel 
**Métrique principale :** AUC-ROC


## Pipeline ML

1. **EDA** — Analyse exploratoire, distribution des classes, corrélations, test Khi-deux, VIF, outliers
1. **Preprocessing** — Correction du data leakage, feature engineering, pipeline sklearn (imputation + encoding + scaling)
1. **Modélisation** — 11 algorithmes testés et comparés par CV Mean AUC
1. **Tuning** — RandomizedSearchCV sur les 3 meilleurs modèles candidats
1. **Évaluation finale** — AUC-ROC, matrice de confusion, feature importance
1. **Déploiement** — Application Streamlit interactive

## Résultats

### Comparaison des modèles (Top 5)

|Modèle           |AUC-ROC   |CV Mean AUC|Diagnostic     |
|-----------------|----------|-----------|---------------|
|**XGBoost**      |**0.8977**|**0.8965** |Bon équilibre  |
|Random Forest    |0.8934    |0.8921     |Légère variance|
|Gradient Boosting|0.8861    |0.8843     |Bon équilibre  |
|Extra Trees      |0.8856    |0.8840     |Légère variance|
|Bagging          |0.8802    |0.8795     |Légère variance|

### Modèle final — XGBoost (après tuning)

|Métrique         |Valeur    |
|-----------------|----------|
|AUC-ROC Test     |**0.9014**|
|AUC-ROC Train    |0.9236    |
|Accuracy         |90.2%     |
|F1-Score (défaut)|0.7793    |

### Matrice de confusion

|               |Prédit Normal|Prédit Défaut|
|---------------|-------------|-------------|
|**Réel Normal**|22 087       |319          |
|**Réel Défaut**|2 601        |4 727        |

- 4 727 défauts détectés sur 7 328
- 319 fausses alarmes sur 22 406 clients sains

## Application Streamlit

L’application permet de :

- Saisir le profil complet d’un dossier emprunteur
- Obtenir la probabilité de défaut en temps réel via une jauge visuelle
- Visualiser les facteurs de risque déclenchés (LTV, DTI, Credit Score, taux)
- Scorer un fichier csv existant

## Structure du projet

```
loan-default-prediction/
├── notebook_FN.ipynb   # Pipeline ML complet avec commentaires
├── app.py    # Application Streamlit
├── model_pipeline.pkl  # Pipeline sérialisé (preprocessing + XGBoost)
├── feature_info.pkl # Métadonnées du modèle
├── rapport_projet_FN.pdf # Rapport complet du projet
├── requirements.txt  # Dépendances Python
└── README.md
```

## Bibliothèques

- **ML :** scikit-learn, XGBoost
- **Visualisation :** Plotly, Matplotlib, Seaborn
- **Déploiement :** Streamlit Cloud
- **Sérialisation :** joblib

## Auteure

**FARES Nouhaila** — MSDE (Edition 7) — EHTP  
Projet Machine Learning 
