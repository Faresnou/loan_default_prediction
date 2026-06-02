import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du défaut de remboursement",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ici on injecte du code HTML/CSS pour modifier l'apparence par défaut de streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC;
    color: #0F172A;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: white !important;
    font-weight: 700;
}

[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-weight: 600;
    font-size: 0.85rem !important;
    margin-top: 10px;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #1E293B !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}

/* Cartes KPI (Conservées intactes) */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.03), 0 1px 3px rgba(15, 23, 42, 0.02);
    border: 1px solid #E2E8F0;
    border-top: 4px solid #2563EB;
}

.kpi-title {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    margin-top: 6px;
    color: #0F172A;
}

/* Boutons */
div.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100%;
    box-shadow: 0 10px 20px rgba(37,99,235,0.2);
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(37,99,235,0.3);
}

/* Tables stylisation */
.stDataFrame {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource # permet de charger le modèle une seule fois au démarrage
def load_model():
    pipeline = joblib.load("model_pipeline.pkl") #Charge le modèle de prédiction XGBoost ainsi que toutes les étapes de préparation des données associées.
    feature_info = joblib.load("feature_info.pkl")#charge le fichier contenant les métadonnées du modèle
    return pipeline, feature_info

try: #structure de sécurité qui permet de bloquer l'éxecution de la page si les fichiers pkl sont introuvables
    pipeline, feature_info = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# En-tête de l'application avec image
st.markdown('<div class="crop-banner">', unsafe_allow_html=True)
st.image("credit.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<style>
.crop-banner, 
.crop-banner data-testid="stImage", 
.crop-banner [data-testid="stImage"] img {
    height: 50px !important; 
    object-fit: cover;        
    object-position: center;
}

.crop-banner {
    overflow: hidden;
    border-radius: 16px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("PRÉDICTION DU DÉFAUT DE REMBOURSEMENT D'UN PRÊT IMMOBILIER (MORTGAGE)")
st.caption("Plateforme d'Analyse et de Prédiction des Risques de Défaut de remboursement de crédits")
st.markdown("---")

# Performances du modèle choisi
st.subheader("Indicateurs de Performance du Modèle")

auc_score = feature_info.get('auc_roc_test', 0.90)
f1_score_ = feature_info.get('f1_test', 0.78)
acc_score = feature_info.get('accuracy_test', 0.90)

m1, m2, m3, m4 = st.columns(4) #pour diviser la page en 4 colonnes 
#on insère un bloc html pour construire les  kpi des performances
with m1:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: #0F172A;">
        <div class="kpi-title">Algorithme</div>
        <div class="kpi-value" style="font-size: 1.6rem; padding-bottom:4px;">XGBoost</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: #10B981;">
        <div class="kpi-title">Modèle AUC-ROC</div>
        <div class="kpi-value">{auc_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: #F59E0B;">
        <div class="kpi-title">F1-Score</div>
        <div class="kpi-value">{f1_score_:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: #6366F1;">
        <div class="kpi-title">Précision (Accuracy)</div>
        <div class="kpi-value">{acc_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# dictionnaires des abréviations des variables expliquées selon les termes juridiques et financiers utilisés dans de tels dataset
loan_purpose_map = {
    "p1": "Achat Résidence Principale",
    "p2": "Refinancement de Crédit",
    "p3": "Investissement Locatif",
    "p4": "Prêt Commercial"
}

credit_worthiness_map = {
    "l1": "Excellente Solvabilité",
    "l2": "Solvabilité Limitée"
}

loan_limit_map = {
    "cf": "Prêt Réglementé (Conforming)",
    "ncf": "Prêt Non-Réglementé (Non-Conforming)"
}

# sidebar (profil client)

st.sidebar.markdown("""
<div style="margin-bottom: 15px;">
    <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
        Section Analyse
    </span>
    <h2 style="color: white; margin: 0; font-size: 1.6rem; font-weight: 800; font-family: 'Inter', sans-serif;">
        Profil du Demandeur
    </h2>
    <hr style="border: 0; border-top: 1px solid #1E293B; margin-top: 10px; margin-bottom: 5px;">
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("### Données Financières")

loan_amount = st.sidebar.number_input("Montant du prêt (Loan Amount)", min_value=0.0, value=150000.0, step=1000.0) 
#crée un champ de saisie  numérique avec min value 0 pour empêcher valeurs négatives et valeur initiale affichée est 150000
#les boutons + et - varie la valeur de 1000 en 1000
#idem pour la suite des variables
property_value = st.sidebar.number_input("Valeur du bien (Property Value)", min_value=0.0, value=250000.0, step=1000.0)
income = st.sidebar.number_input("Revenu annuel (Annual Income)", min_value=0.0, value=85000.0, step=1000.0)
credit_score = st.sidebar.number_input("Score de crédit (Credit Score)", min_value=300, max_value=900, value=700)
term = st.sidebar.number_input("Durée du prêt en mois (Loan Term)", min_value=12, max_value=480, value=360)
dtir1 = st.sidebar.slider("Ratio d'endettement (Debt-To-Income Ratio)", min_value=0, max_value=100, value=35)#barre de défilement horizontale de 0 à 100%

st.sidebar.markdown("### Caractéristiques & Spécificités")
loan_limit = st.sidebar.selectbox("Limite du prêt (Loan Limit)", options=list(loan_limit_map.keys()), format_func=lambda x: loan_limit_map[x]) 
#crée un menu déroulant avec une fonction qui prend la clé technique et la remplace visuellement à l'écran par sa valeur textuelle propre
#idem pour la suite
gender = st.sidebar.selectbox("Genre (Gender)", ["Male", "Female", "Joint", "Sex Not Available"])
approv_in_adv = st.sidebar.selectbox("Approbation préalable (Approval in Advance)", ["pre", "nopre"])
loan_type = st.sidebar.selectbox("Type de prêt (Loan Type)", ["type1", "type2", "type3"])
loan_purpose = st.sidebar.selectbox("Objet du prêt (Loan Purpose)", options=list(loan_purpose_map.keys()), format_func=lambda x: loan_purpose_map[x])
credit_worthiness = st.sidebar.selectbox("Niveau de solvabilité (Credit Worthiness)", options=list(credit_worthiness_map.keys()), format_func=lambda x: credit_worthiness_map[x])
open_credit = st.sidebar.selectbox("Crédit permanent ouvert (Open Credit)", ["opc", "nopc"])
business_or_commercial = st.sidebar.selectbox("Usage commercial (Business or Commercial)", ["b/c", "nob/c"])
neg_ammortization = st.sidebar.selectbox("Amortissement négatif (Negative Amortization)", ["neg_amm", "not_neg"])
interest_only = st.sidebar.selectbox("Intérêts uniquement (Interest Only)", ["int_only", "not_int"])
lump_sum_payment = st.sidebar.selectbox("Remboursement unique in fine (Lump Sum Payment)", ["lpsm", "not_lpsm"])
construction_type = st.sidebar.selectbox("Type de construction (Construction Type)", ["sb", "mh"])
occupancy_type = st.sidebar.selectbox("Type d'occupation du bien (Occupancy Type)", ["pr", "sr", "ir"])
secured_by = st.sidebar.selectbox("Type de garantie (Secured By)", ["home", "land"])
total_units = st.sidebar.selectbox("Nombre total de logements (Total Units)", ["1U", "2U", "3U", "4U"])
credit_type = st.sidebar.selectbox("Organisme de crédit (Credit Type)", ["EXP", "EQUI", "CRIF", "CIB"])
co_applicant_credit_type = st.sidebar.selectbox("Type crédit co-emprunteur (Co-Applicant Credit Type)", ["CIB", "EXP"])
age = st.sidebar.selectbox("Tranche d'âge (Age Range)", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"])
submission_of_application = st.sidebar.selectbox("Canal de soumission (Submission of Application)", ["to_inst", "not_inst"])
region = st.sidebar.selectbox("Région géographique (Region)", ["North", "North-East", "central", "south"])
security_type = st.sidebar.selectbox("Type de sûreté (Security Type)", ["direct", "Indriect"])

# EATURE ENGINEERING
LTV = (loan_amount / property_value * 100) if property_value > 0 else 0
#montant du prêt divisé par la valeur du bien

# résumé dynamique des données saisies par l'utilisateur
st.subheader("Caractéristiques du Dossier Actuel")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Montant Demandé</div>
        <div class="kpi-value">${loan_amount:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Valeur Estimée du Bien</div>
        <div class="kpi-value">${property_value:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Score de Crédit Client</div>
        <div class="kpi-value">{credit_score}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Ratio LTV Calculé</div>
        <div class="kpi-value">{LTV:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


# créatio  du bloc de prédiction
st.subheader("Analyse et prédiction du défaut de remboursement")

predict_button = st.button("Lancer l'Évaluation du Risque de Crédit")

if predict_button: #si on clique sur le bouton le code s'éxecute
    input_data = { #récupère l'intégralité des variables saisies dans la sidebar et les transforme en dictionnaire
        'year': 2019, 'loan_limit': loan_limit, 'Gender': gender, 'approv_in_adv': approv_in_adv,
        'loan_type': loan_type, 'loan_purpose': loan_purpose, 'Credit_Worthiness': credit_worthiness,
        'open_credit': open_credit, 'business_or_commercial': business_or_commercial, 'loan_amount': loan_amount,
        'term': term, 'Neg_ammortization': neg_ammortization,
        'interest_only': interest_only, 'lump_sum_payment': lump_sum_payment, 'property_value': property_value,
        'construction_type': construction_type, 'occupancy_type': occupancy_type, 'Secured_by': secured_by,
        'total_units': total_units, 'income': income, 'credit_type': credit_type, 'Credit_Score': credit_score,
        'co-applicant_credit_type': co_applicant_credit_type, 'age': age, 'submission_of_application': submission_of_application,
        'LTV': LTV, 'Region': region, 'Security_Type': security_type, 'dtir1': dtir1
    }

    df_input = pd.DataFrame([input_data])

    try: #ici on ordonne les caractéristiques dans le même ordre que dans l'entraînement
        expected_features = feature_info['all_features']
        for col in expected_features:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_input = df_input[expected_features]
    except:
        pass

    try:
        prediction = pipeline.predict(df_input)[0]
        probability = pipeline.predict_proba(df_input)[0][1]
        risk_percent = probability * 100

        # code de la jauge visuelle
        # Détermination de la couleur principale selon le risque
        gauge_color = "#EF4444" if risk_percent >= 50 else "#10B981"
        status_title = "DOSSIER À RISQUE ÉLEVÉ (REJETÉ)" if risk_percent >= 50 else "DOSSIER VALIDÉ (RISQUE FAIBLE)"

        # Création de la jauge avec Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 48, 'color': '#0F172A', 'family': 'Inter'}},
            title = {'text': status_title, 'font': {'size': 20, 'color': gauge_color, 'weight': 'bold', 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                'bar': {'color': gauge_color}, # Couleur du curseur/remplissage
                'bgcolor': "#E2E8F0",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},  # Zone Verte (Risque faible)
                    {'range': [30, 50], 'color': 'rgba(245, 158, 11, 0.1)'}, # Zone Orange (Risque modéré)
                    {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.1)'}  # Zone Rouge (Risque élevé)
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 50 # Ligne rouge critique à 50%
                }
            }
        ))
        # Ajustement des marges
        fig.update_layout(
            height=250, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)', # Fond transparent pour s'adapter à Streamlit
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Affichage du composant dans un conteneur Streamlit
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            #ici on ajoute le niveau de confiance (probabilité)
            confidence_percent = 100.0 - risk_percent
            # Affichage du Niveau de Confiance 
            st.markdown(f"""
            <div style="text-align: center; padding: 12px; margin-top: 5px; margin-bottom: 15px; border-radius: 8px; background-color: {gauge_color}10; border: 1px solid {gauge_color}30;">
                <p style="margin: 0; color: #64748B; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Niveau de Confiance de l'Accord
                </p>
                <p style="margin: 5px 0 0 0; color: #0F172A; font-size: 2.2rem; font-weight: 800; font-family: 'Inter', sans-serif;">
                    {confidence_percent:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            # Message de conclusion sous la jauge
            if risk_percent >= 50:
                st.warning("Attention : L'application présente un risque de remboursement hors limites réglementaires.")
            else:
                st.success("Félicitations : Le profil du demandeur répond positivement aux critères de prêt")

         # facteurs explicatifs du rejet du dossier
        insights = [] # les valeurs choisies correspondent aux seuils d'alerte universels appliqués dans l'immobilier et le crédit
        if credit_score < 650: #mauvais payeur potentiel
            insights.append("Le faible score de crédit (Credit Score) dégrade la note globale.")
        if LTV > 80: #le ratio entre le montant du prêt et la valeur de la maison
            insights.append("Le ratio d'emprunt sur la valeur du bien (LTV) est excessivement haut.")
        if dtir1 > 45: #C'est le taux d'endettement (les mensualités par rapport aux revenus)
            insights.append("Le taux d'endettement de l'acheteur (Debt-To-Income) dépasse le seuil d'alerte.")
        
        if len(insights) > 0:
            st.markdown("### Principaux Facteurs de Risque Détectés")
            for item in insights:
                st.write(f"• {item}")

    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la prédiction : {e}")
    
# partie pour charger un fichier csv depuis l'ordinateur
st.subheader("Traitement de Fichiers CSV")

uploaded_file = st.file_uploader("Importer un fichier client au format CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("**Aperçu des données importées :**")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Lancer le Scoring Global du fichier"): #crée un bouton pour le traitement du fichier
            expected_features = feature_info['all_features']
# ici on répète la même procédure d'alignement, d'ajustement et de vérification des colonnes que celle effectuée pour la prédiction individuelle
            for col in expected_features:
                if col not in batch_df.columns:
                    batch_df[col] = np.nan

            batch_df = batch_df[expected_features]

            predictions = pipeline.predict(batch_df)
            probabilities = pipeline.predict_proba(batch_df)[:, 1] #sert à extraire la colonne des probabilités de la classe 1

            results_df = batch_df.copy()
            results_df['Prediction'] = predictions
            results_df['Default_Probability'] = probabilities

            st.success("Analyse groupée exécutée avec succès.")
            st.dataframe(results_df.head(), use_container_width=True)

            csv = results_df.to_csv(index=False).encode('utf-8') #convertit les résultats en csv

            st.download_button(
                label="Télécharger le Rapport Complet Scoré", #crée un bouton de télechargement
                data=csv,
                file_name="credit_scoring_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Erreur lors de l'analyse du fichier CSV : {e}")