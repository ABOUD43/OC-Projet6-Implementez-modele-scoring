import streamlit as st #data web app development
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.cm import RdYlGn
import requests
import numpy as np
import math
import seaborn as sns
#from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import pickle
import plotly.express as px #interactive charts
import shap 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
from sklearn.preprocessing import StandardScaler
import joblib


# URL de l'API Flask (local)
#API_URL = "http://127.0.0.1:5000"

# URL de l'API Flask (distant)
API_URL = "https://gentle-sea-64911-1faf4d117eb6.herokuapp.com"



def get_prediction_from_api(client_id):
    try:
        # Faire une requête GET à l'API Flask pour obtenir les informations de prédiction
        response = requests.get(f"{API_URL}/{client_id}/")
        
        # Vérifier si la requête a réussi
        if response.status_code == 200:
            return response.text  # Le texte renvoyé par l'API, qui contient les informations de solvabilité
        else:
            return f"Erreur lors de la récupération de la prédiction: {response.status_code}"
    
    except Exception as e:
        return f"Erreur: {str(e)}"


st.set_page_config(
    page_title="Pret a DEPENSER Dashboard",
    page_icon="✅",
    layout="wide",) #wide-screen

style = """
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .small-font {
        font-size:10px !important;
    }
</style>
"""

# Injectez le CSS avec Markdown
st.markdown(style, unsafe_allow_html=True)


@st.cache #mise en cache de la fonction pour exécution unique

def load_data():

    dataframe = pd.read_csv("dataframe.csv")

    predi = pd.read_csv("prediction.csv")
    Xtest= pd.read_csv('Xtest.csv')

    with open("explainer_lgbm.pkl", 'rb') as explainer_file:
        explainer = pickle.load(explainer_file)

    with open("model_lgbm.pickle", 'rb') as file: 
        clf_lgbm = pickle.load(file) 
    
    customer = dataframe['SK_ID_CURR']
    customer=customer.astype('int64')
  
    with open('scaler.pkl', 'rb') as f:
        std = pickle.load(f)

    with open('imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    
   
    

    return dataframe , customer, predi  , clf_lgbm , explainer,Xtest,std,imputer

    
dataframe ,customer, predi ,clf_lgbm , explainer,Xtest, std, imputer = load_data()

st.markdown('<p class="big-font">Prêt à DEPENSER</p>', unsafe_allow_html=True)		

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<p class="medium-font">Prédictions de la capacité d\'un client à rembourser son prêt</p>', unsafe_allow_html=True)
customer_ids = predi['SK_ID_CURR'].unique()
customer_ids = customer_ids.astype(str) 
id_client = st.selectbox('Veuillez choisir l\'identifiant du client:', ['Choisir un ID'] + list(customer_ids))

if id_client == 'Choisir un ID':
    st.error('Merci de choisir un ID client dans la liste.')

	
else:
    id_client = int(id_client)
    with st.spinner('Chargement du score du client...'):

        st.header("**Analyse dossier client**")
        result = get_prediction_from_api(id_client)
        st.write(result)

        class_cust = int(predi[predi['SK_ID_CURR']==int(id_client)]['predict'].values[0])
        proba = predi[predi['SK_ID_CURR']==int(id_client)]['proba'].values[0]
        classe_vrai = int(predi[predi['SK_ID_CURR']==int(id_client)]['TARGET'].values[0])

        if class_cust == 1: 
           
            reponse= "Prêt non accordé"
            
            st.markdown('<style>p{color: orange;}</style>', unsafe_allow_html=True)
        else:
        
            reponse= 'Prêt accordé'


        list_infos = ['SK_ID_CURR', 'NAME_FAMILY_STATUS','CODE_GENDER', 'DAYS_BIRTH', 'CNT_CHILDREN','NAME_CONTRACT_TYPE','AMT_INCOME_TOTAL',
                    'AMT_GOODS_PRICE','AMT_CREDIT', 'DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2',
        'EXT_SOURCE_3','AMT_ANNUITY' ,'FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE' ]

        df_clients = dataframe[dataframe['SK_ID_CURR'] == int(id_client)]
        df_clients = df_clients.loc[:,list_infos] 
        df_clients.index = ['Information']       
        df_clients = df_clients.rename(columns={'SK_ID_CURR': 'Numero client'})
        df_clients["Age"]= int(df_clients["DAYS_BIRTH"]/-365) #.values
        df_clients['DAYS_EMPLOYED'] = - df_clients['DAYS_EMPLOYED']	
        df_clients = df_clients.rename(columns={'CODE_GENDER': 'Genre'})   
        df_clients = df_clients.rename(columns={'FLAG_OWN_REALTY': 'APPT'})
        df_clients = df_clients.rename(columns={'FLAG_OWN_CAR': 'Voiture'})
        df_clients = df_clients.rename(columns={'NAME_INCOME_TYPE' : 'Type emploi'})
        df_clients = df_clients.rename(columns={'': ''})
        df_clients = df_clients.rename(columns={'AMT_INCOME_TOTAL': 'Total revenus du Client'})
        df_clients = df_clients.rename(columns={'AMT_CREDIT': 'Montant du crédit'})
        df_clients = df_clients.rename(columns={'AMT_ANNUITY': 'Annuités crédit'})
        df_clients = df_clients.rename(columns={'AMT_GOODS_PRICE': 'Montant du bien pour le crédit'})
        df_clients = df_clients.rename(columns={'NAME_FAMILY_STATUS': 'Statut famille'})
        df_clients = df_clients.rename(columns={'CNT_CHILDREN': 'Nombre enfant(s)'})
        df_clients= df_clients.drop('DAYS_BIRTH', axis=1) 
        df_clients = df_clients.rename(columns={'DAYS_EMPLOYED': 'Nbr de jours travaillés'})
        df_clients = df_clients.rename(columns={'NAME_CONTRACT_TYPE': 'Type de prêt'})
        st.success('Profil du client: ')
        st.dataframe(df_clients[['Numero client',"Age","Genre",'Statut famille','Nombre enfant(s)','APPT','Voiture','Type emploi',
        'Nbr de jours travaillés','Total revenus du Client']])
        st.success('Détails du prêt')
        st.table(df_clients[['Type de prêt', 'Montant du crédit', 'Annuités crédit', 'Montant du bien pour le crédit']])   









        threshold_value = 0.478
        

        kpi1, kpi2 = st.columns(2) 
        kpi1.metric(label="Probabilité de défaut de paiement",value=f"{round(proba*100):.2f}%", 
                    delta=None, delta_color="normal", help=None)  
        kpi2.metric(label="Décision",value=reponse, delta=None, delta_color="normal", help=None)

        gauge_predict = go.Figure(go.Indicator(mode="gauge+number",value=proba,number={'valueformat': '.3f', 'font': {'size': 36}}, 

        title={'text': f"Client {id_client}", 'font': {'size': 20}}, domain={'x': [0, 1], 'y': [0, 1]},

        gauge={
        'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': "darkblue", 'nticks': 10},  
        'bar': {'color': "darkblue", 'thickness': 0.3},  # Couleur de l'aiguille
        'bgcolor': "white",  
        'steps':[
                    {'range': [0, 0.478], 'color': 'lightgreen'},
                    {'range': [0.478, 1], 'color': 'lightcoral'}],
        'threshold': {
            'line': {'color': "red", 'width': 6},  
            'thickness': 0.75,
            'value': threshold_value}}  ))

        gauge_predict.update_layout(
        margin={'t': 50, 'b': 50, 'l': 50, 'r': 50}, 
        height=400,  
        paper_bgcolor="lightgray", 
        font={'color': "darkblue", 'family': "Arial"} ) 
        st.plotly_chart(gauge_predict, use_container_width=False)
        st.markdown('<p style="text-align:center;color:grey;">Ce score indique la probabilité de défaut de paiement pour le client sélectionné. Le seuil est fixé à 0.478.</p>', unsafe_allow_html=True)
        


      
        #  'anomaly_value' est la valeur d'anomalie du nombre de jours travaillés
        #anomaly_value = 365243
        #if df_clients[df_clients['Nbr de jours travaillés'] == anomaly_value]:
            #st.text("Jours d'emploi: Donnée non standard ou manquante")
            #st.caption("Note: Les valeurs de jours d'emploi exceptionnellement élevées sont considérées comme non standards et peuvent indiquer une retraite ou un emploi non enregistré.")
        #else:
            #st.text(f"Jours d'emploi: {-df_clients['Nbr de jours travaillés']} jours")
# dans dataframe apres traitement on a un colonne dataframe["DAYS_EMPLOYED_ANOM"] pour indiquer si la valeur etait anormale
    

        # Calculer les valeurs SHAP pour le client spécifique
        client_features = Xtest[Xtest['SK_ID_CURR'] == int(id_client)].drop(columns=['TARGET','SK_ID_CURR'], axis=1)
        client_features.drop(client_features.columns[0], axis=1, inplace=True)

        client_features_imputer= imputer.transform(client_features)

        client_features_scaled = std.transform(client_features_imputer)
        new=pd.DataFrame(data=client_features_scaled,columns=client_features.columns)
     
        #Valeurs SHAP pour le client spécifique
        shap_values = explainer.shap_values(new)

        #Tracer la visualisation SHAP pour le client spécifique
        feature_names = list(client_features.columns)
       
        #Créer un DataFrame avec les valeurs SHAP et les noms des caractéristiques
        df_shap= pd.DataFrame({'feature': feature_names, 'SHAP value': shap_values[1][0]})

        # Trier les valeurs SHAP par ordre décroissant
        df_shap.sort_values('SHAP value', ascending=False, inplace=True)
        dfshap = pd.concat([df_shap.nlargest(5, 'SHAP value'), df_shap.nsmallest(5, 'SHAP value')])
        colors = [RdYlGn(0.05*i) for i in range(5)] + [RdYlGn(0.8 + 0.04*i) for i in range(5)]
         
         # Afficher le graphique avec les noms des caractéristiques
        st.markdown("""
    <h1 style='text-align: center; font-size: 24px;'>Importance des Caractéristiques</h1>
    <h2 style='text-align: center; font-size: 18px;'>Les 5 contributions les plus positives et les 5 contributions les plus négatives sur la probabilité de défaut de paiement</h2>
    """, unsafe_allow_html=True)
        fig = px.bar(dfshap, y='feature', x='SHAP value', orientation='h', color=colors)
        fig.update_layout(width= 600 ,height=500, xaxis_title="'Valeur SHAP",yaxis_title="Caractéristique",showlegend=False)#fond du graphique transparent plot_bgcolor='rgba(0,0,0,0)'
        #fig.update_traces(texttemplate='%{x}')#textposition='outside'
        st.plotly_chart(fig, use_container_width=True)



     
        mask_1 = (dataframe['TARGET'] == 1)
        mask_0 = (dataframe['TARGET'] == 0)
        data= dataframe.replace(np.nan,0)
      
        data_1 = [data.loc[mask_0,'EXT_SOURCE_1'],data.loc[mask_1,'EXT_SOURCE_1']]
        data_2 = [data.loc[mask_0,'EXT_SOURCE_2'],data.loc[mask_1,'EXT_SOURCE_2']]
        data_3 = [data.loc[mask_0,'EXT_SOURCE_3'],data.loc[mask_1,'EXT_SOURCE_3']]
        data_4 = [data.loc[mask_0,'AMT_CREDIT'],data.loc[mask_1,'AMT_CREDIT']]
        # Définir les groupes et les couleurs

        colors = ['#37AA9C','#EB89B5']
        import plotly.graph_objs as go
        group_labels = ['TARGET 0', 'TARGET 1']

        st.subheader('Historique d\'emploi')
        non_anomalo_data = dataframe[dataframe['DAYS_EMPLOYED'] != 365243]
        non_anomalo_data['DAYS_EMPLOYED'] = non_anomalo_data['DAYS_EMPLOYED'].abs()
      
        fig = px.histogram(non_anomalo_data, x='DAYS_EMPLOYED',
                   title="Distribution du nombre de jours d'emploi des clients (sans anomalies)",
                   labels={'DAYS_EMPLOYED': 'Nombre de jours'}, 
                   opacity=0.8,
                   color_discrete_sequence=['skyblue'])  
        fig.update_layout(width= 500, height=500,xaxis_title='Nombre de jours',yaxis_title='Nombre de clients')
        x_value=df_clients['Nbr de jours travaillés'].iloc[0] #df_clients['Nbr de jours travaillés'].values[0]
        fig.add_trace(go.Scatter(x=[x_value,x_value],y=[0,6000], mode="lines", name='Client', line=go.scatter.Line(color="red")))
        st.plotly_chart(fig,use_container_width=True)
      
        median_income = dataframe['AMT_INCOME_TOTAL'].median() #total revenus de client 
        median_credit = dataframe['AMT_CREDIT'].median() #montant de credit 
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Client',x=df_clients[['Total revenus du Client', 'Montant du crédit']].values.flatten(),y=['Revenus', 'Crédit'],orientation='h'))
        fig.add_trace(go.Bar(name='Médiane Groupe',x=[median_income, median_credit],y=['Revenus', 'Crédit'],orientation='h'))

        fig.update_layout(width= 500, height=500,barmode='group', title='Comparaison des Revenus et du Montant du Crédit avec la Médiane du Groupe de Référence')
        st.plotly_chart(fig,use_container_width=True)

   
        color =['#636EFA', '#EF553B']
        client_color = "red" 
    

       
        sns.set_context("notebook", font_scale=0.8)
        sns.set(rc={'figure.figsize':(5,4)})
        plt.rc('axes', titlesize=3)  
        plt.rc('axes', labelsize=3) 
        plt.rc('xtick', labelsize=3)  
        plt.rc('ytick', labelsize=3)  
        plt.rc('legend', fontsize=3)
        fig, axes = plt.subplots(2, 2)
    
        sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT']
       

        fig = make_subplots(rows=1, cols=3, subplot_titles=("Source Extérieure 1", "Source Extérieure 2", "Source Extérieure 3"))
        fig1= ff.create_distplot(data_1, group_labels, histnorm='',show_hist=False, colors=colors, show_rug=False) 
        fig2= ff.create_distplot(data_2, group_labels, histnorm='', bin_size=0.5, show_hist=False, colors=colors, show_rug=False)
        fig3= ff.create_distplot(data_3, group_labels, histnorm='', bin_size=0.5 , show_hist=False, colors=colors, show_rug=False)
  

        for trace in fig1.select_traces(): # ff.create_distplot retourne un objet Figure complet qui a  de méthode select_traces()
            fig.add_trace(trace, row=1, col=1)

        for trace in fig2.select_traces():
            trace['showlegend'] = False
            fig.add_trace(trace, row=1, col=2)
        
        for trace in fig3.select_traces():
            trace['showlegend'] = False
            fig.add_trace(trace, row=1, col=3)
        

        fig.update_layout(title="Scores de crédit externes",showlegend=True,legend=dict(itemsizing="constant" ))
        fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_1'].iloc[0],df_clients['EXT_SOURCE_1'].iloc[0]], y=[0, 10], mode="lines", name='Client',line=dict(color=client_color, width=4)),row=1,col=1)
        fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_2'].iloc[0], df_clients['EXT_SOURCE_2'].iloc[0]], y=[0, 3], mode="lines", name='Client', line=go.scatter.Line(color=client_color),showlegend=False ), row=1, col=2)
        fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_3'].iloc[0], df_clients['EXT_SOURCE_3'].iloc[0]], y=[0, 3], mode="lines", name='Client', line=go.scatter.Line(color=client_color),showlegend=False ), row=1, col=3)

        fig.update_layout(width=10000, height=500, font=dict(size=15))
        fig.update_xaxes(title_text="Source Ext1", row=1, col=1)
        fig.update_xaxes(title_text="Source Ext2", row=1, col=2)
        fig.update_xaxes(title_text="Source Ext3", row=1, col=3)
       
        st.plotly_chart(fig, use_container_width=True)

    

        fig=make_subplots(rows=1,cols=2, subplot_titles=("Proportion de clients possédant une voiture", "Proportion de clients possédant un bien immobilier"))
        color_map= {0: '#37AA9C', 1: '#EB89B5'} 
        # Graphique pour la possession de voiture

        fig1 = px.histogram(dataframe,x="FLAG_OWN_CAR",color='TARGET',
        labels={'FLAG_OWN_CAR':'Possession de voiture','TARGET':'Statut de paiment'},barmode='group',color_discrete_map=color_map)

        fig2=px.histogram(dataframe,x="FLAG_OWN_REALTY",color='TARGET',
        labels={'FLAG_OWN_REALTY':'Possession immobilière','TARGET':'Statut de paiment'},barmode='group',color_discrete_map=color_map)

        for trace in fig1.select_traces():
            fig.add_trace(trace, row=1, col=1)

        for trace in fig2.select_traces():
            fig.add_trace(trace,row=1,col=2)
        

        fig.update_traces(showlegend=True)

        fig.update_layout(barmode='group')
        #st.plotly_chart(fig,use_container_width=True)



#Déterminer les individus les plus proches du client dont l'id est séléctionné
check_voisins = st.checkbox("Afficher dossiers similaires?")



if check_voisins:
    # Calculer l'âge en années
    dataframe['age'] = -(dataframe["DAYS_BIRTH"] / 365)
    
    # Récupérer les informations du client étudié
    job = dataframe[dataframe["SK_ID_CURR"] == int(id_client)][['AMT_CREDIT', 'age', 'CODE_GENDER', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','EXT_SOURCE_1',
    'EXT_SOURCE_2','EXT_SOURCE_3','AMT_GOODS_PRICE']]

    job['DAYS_EMPLOYED'] = abs(job['DAYS_EMPLOYED'].values[0])
    
    # Filtrer les données pour trouver les clients avec des caractéristiques similaires : gendre/age/montant de credit/jours d'emploi/revenu total/montant de crédit/montant de bien
    ##df_voisin = dataframe[dataframe['CODE_GENDER'] == job.at[job.index[0], 'CODE_GENDER']]
    
    # Remplacer les NaN par 0 pour éviter des erreurs lors des comparaisons
    df_voisin = dataframe.replace(np.nan, 0)#df_voisin

    df_voisin['DAYS_EMPLOYED'] = df_voisin['DAYS_EMPLOYED'].apply(lambda x: abs(x))
   
    
    # Filtrage basé sur l'âge (proximité de 5 ans)
    df_voisin = df_voisin[(df_voisin['age'] < (job.at[job.index[0], 'age'] + 5)) &
                          (df_voisin['age'] > (job.at[job.index[0], 'age'] - 5))]
    
    # Filtrage basé sur le montant du crédit (proximité de 20%)
    df_voisin = df_voisin[((df_voisin['AMT_CREDIT'] / job.at[job.index[0], 'AMT_CREDIT']) <= 1.2) &
                          ((df_voisin['AMT_CREDIT'] / job.at[job.index[0], 'AMT_CREDIT']) >= 0.8)]
    
    # Filtrage basé sur le nombre de jours d'emploi (proximité de 2 ans)
    df_voisin = df_voisin[(df_voisin['DAYS_EMPLOYED'] < (job.at[job.index[0], 'DAYS_EMPLOYED'] + 730)) &
                          (df_voisin['DAYS_EMPLOYED'] > (job.at[job.index[0], 'DAYS_EMPLOYED'] - 730))]
    
    # Filtrage basé sur le revenu total (proximité de 20%)
    df_voisin = df_voisin[((df_voisin['AMT_INCOME_TOTAL'] / job.at[job.index[0], 'AMT_INCOME_TOTAL']) <= 1.2) &
                          ((df_voisin['AMT_INCOME_TOTAL'] / job.at[job.index[0], 'AMT_INCOME_TOTAL']) >= 0.8)]

    df_voisin['CREDIT_INCOME_RATIO'] = df_voisin['AMT_CREDIT'] / df_voisin['AMT_INCOME_TOTAL']

    #Filtrer les voisins en fonction du montant de bien proche (±20%)
    df_voisin_proche_bien= df_voisin[((df_voisin['AMT_GOODS_PRICE'] / job.at[job.index[0], 'AMT_GOODS_PRICE']) <= 1.2) &
                      ((df_voisin['AMT_GOODS_PRICE'] / job.at[job.index[0], 'AMT_GOODS_PRICE']) >= 0.8)]

    df_voisin_proche_bien['CREDIT_INCOME_RATIO'] = df_voisin_proche_bien['AMT_CREDIT'] / df_voisin_proche_bien['AMT_INCOME_TOTAL']




    client_target = class_cust
    st.write(f"Nombre de clients voisins trouvés: {len(df_voisin)}")

    fig_ratio = px.scatter(df_voisin,
                       x='CREDIT_INCOME_RATIO',
                       y='AMT_GOODS_PRICE',
                       color='TARGET', 
                       hover_data=['CREDIT_INCOME_RATIO', 'AMT_GOODS_PRICE', 'TARGET'],
                       labels={'CREDIT_INCOME_RATIO': 'Ratio Crédit / Revenu', 'AMT_GOODS_PRICE': 'Montant du Bien'},
                       title='Comparaison du Ratio Crédit/Revenu et Montant du Bien pour les Clients Voisins',
                       color_continuous_scale=None)
    
                       
    fig_ratio.add_trace(go.Scatter(
    x=[job['AMT_CREDIT'].values[0] / job['AMT_INCOME_TOTAL'].values[0]],
    y=[job['AMT_GOODS_PRICE'].values[0]],
    mode='markers+text',
    marker=dict(color='red', size=15, symbol='star'),
    name='Client étudié',
    hovertext=[f"Ratio Crédit / Revenu: {job['AMT_CREDIT'].values[0] / job['AMT_INCOME_TOTAL'].values[0]:.2f}<br>"
               f"Montant du Bien: {job['AMT_GOODS_PRICE'].values[0]}<br>"
               f"TARGET={client_target}"]))
        
    #text=[f"Client étudié (TARGET={client_target})"],
    #textposition='top center',))

    st.plotly_chart(fig_ratio)  

    fig_ratio_proche_bien = px.scatter(df_voisin_proche_bien,
                                   x='CREDIT_INCOME_RATIO',
                                   y='AMT_GOODS_PRICE',
                                   color='TARGET',  
                                   hover_data=['CREDIT_INCOME_RATIO', 'AMT_GOODS_PRICE', 'TARGET'],
                                   labels={'CREDIT_INCOME_RATIO': 'Ratio Crédit / Revenu', 'AMT_GOODS_PRICE': 'Montant du Bien'},
                                   title='Comparaison du Ratio Crédit/Revenu et Montant du Bien pour les Clients Voisins (Bien Proche)')

    fig_ratio_proche_bien.add_trace(go.Scatter(
    x=[job['AMT_CREDIT'].values[0] / job['AMT_INCOME_TOTAL'].values[0]],
    y=[job['AMT_GOODS_PRICE'].values[0]],
    mode='markers+text',
    marker=dict(color='red', size=15, symbol='star'),
    name='Client étudié',
    text=[f"Client étudié (TARGET={client_target})"],
    textposition='top center',))

    #st.plotly_chart(fig_ratio_proche_bien)


    fig = px.scatter(df_voisin, 
                 x='AMT_CREDIT', 
                 y='AMT_INCOME_TOTAL', 
                 color='TARGET',
                 color_discrete_map={0: 'blue', 1: 'yellow'},
                 hover_data=['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'TARGET'],
                 labels={'AMT_CREDIT': 'Montant du Crédit', 'AMT_INCOME_TOTAL': 'Revenu Total'},
                 title='Comparaison du Client avec les Clients Voisins')
    fig.add_trace(go.Scatter(
    x=[job['AMT_CREDIT'].values[0]], 
    y=[job['AMT_INCOME_TOTAL'].values[0]],
    mode='markers+text',
    marker=dict(color='red', size=15, symbol='star'),
    name='Client étudié',
    text=["Client étudié (TARGET=1)"],
    textposition='top center',))


    #st.plotly_chart(fig)
    #st.write(df_voisin.head())
    #st.write(job)
    #st.write(f"Nombre de clients voisins trouvés: {len(df_voisin)}")


# Filtrage par target pour les clients voisins
    mask_target1 = (df_voisin['TARGET'] == 1)
    mask_target0 = (df_voisin['TARGET'] == 0)

# Récupération des données à tracer
    data_source1 = [df_voisin.loc[mask_target0, 'EXT_SOURCE_1'], df_voisin.loc[mask_target1, 'EXT_SOURCE_1']]
    data_source2 = [df_voisin.loc[mask_target0, 'EXT_SOURCE_2'], df_voisin.loc[mask_target1, 'EXT_SOURCE_2']]
    data_source3 = [df_voisin.loc[mask_target0, 'EXT_SOURCE_3'], df_voisin.loc[mask_target1, 'EXT_SOURCE_3']]
  

    group_labels = ['Non Défaillant', 'Défaillant']
    colors = ['#37AA9C', '#EB89B5']

# Création des subplots pour les 4 distributions
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Source Extérieure 1", "Source Extérieure 2", "Source Extérieure 3"))

# Tracer les distributions avec une transparence et des légendes
    fig1 = ff.create_distplot(data_source1, group_labels, show_hist=False, colors=colors, curve_type='kde')
    fig2 = ff.create_distplot(data_source2, group_labels, show_hist=False, colors=colors, curve_type='kde')
    fig3 = ff.create_distplot(data_source3, group_labels, show_hist=False, colors=colors, curve_type='kde')
   

# Ajouter les traces aux subplots
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2['data']:
        trace['showlegend'] = False
        fig.add_trace(trace, row=1, col=2)

    for trace in fig3['data']:
        trace['showlegend'] = False
        fig.add_trace(trace, row=1, col=3)

  

# Mettre à jour le layout pour plus de clarté
    fig.update_layout(width=1000, height=500,font=dict(size=15, color="#7f7f7f"),title_text="Comparaison des sources externes",showlegend=True)


    fig.update_xaxes(title_text="Source Ext1", row=1, col=1)
    fig.update_xaxes(title_text="Source Ext2", row=1, col=2)
    fig.update_xaxes(title_text="Source Ext3", row=1, col=3)


# Ajout des lignes du client
    client_color = "red"
    fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_1'].iloc[0], df_clients['EXT_SOURCE_1'].iloc[0]], 
                         y=[0, 5], mode="lines", name='Client (S Ext1)', line=dict(color=client_color, dash='dash', width=3)), 
              row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_2'].iloc[0], df_clients['EXT_SOURCE_2'].iloc[0]], 
                         y=[0, 3], mode="lines", name='Client (S Ext2)', line=dict(color=client_color, dash='dash', width=3),showlegend=False), 
              row=1, col=2)
    fig.add_trace(go.Scatter(x=[df_clients['EXT_SOURCE_3'].iloc[0], df_clients['EXT_SOURCE_3'].iloc[0]], 
                         y=[0, 3], mode="lines", name='Client (S Ext3)', line=dict(color=client_color, dash='dash', width=3),showlegend=False), 
              row=1, col=3)


# Affichage du graphique
    st.plotly_chart(fig)






     
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)