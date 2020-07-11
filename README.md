# P7 Openclassroom :  Implémentez un modèle de scoring
### Openclassroom Datascience training by Salaheddine EG

## Structure du projet : 
- input : fichiers CSV de données téléchargés de Kaggle
- notebooks : Notebooks de nettoyage et de modélisation
  - cleaning : Notebooks de nettoyage adaptés depuis les notebooks sur https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering, https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2 et https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection
  - modelisation :
    - P7_modelisation : Notebook de modélisation, et de définition de la fonction d'évaluation
	- P7_Data_Preparation_For_Dashboard : manipulations en vue de préparer le cache de données pour le dashboard
- output : Sorties de modélisation / nettoyage / caching 
  - dashboard : les  fichiers utilisés par le backend du dashboard
- web : code pour générer le backend et le dashboard
  - app.py : backend Flask API
  - dashboard.py : Front end utilisant Streamlit


## Liens pour le dashboard : 
Le dashboard a été deployé sur Heroku, pour y accéder : 
http://pret-a-depenser.herokuapp.com/
le backend aussi a été deployé sur Heroku, exemple d'appel : 
https://pret-a-depenser-backend.herokuapp.com/get_all_clients

