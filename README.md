# P7 Openclassroom :  Impl�mentez un mod�le de scoring
### Openclassroom Datascience training by Salaheddine EG

## Structure du projet : 
- input : fichiers CSV de donn�es t�l�charg�s de Kaggle
- notebooks : Notebooks de nettoyage et de mod�lisation
  - cleaning : Notebooks de nettoyage adapt�s depuis les notebooks sur https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering, https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2 et https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection
  - modelisation :
    - P7_modelisation : Notebook de mod�lisation, et de d�finition de la fonction d'�valuation
	- P7_Data_Preparation_For_Dashboard : manipulations en vue de pr�parer le cache de donn�es pour le dashboard
- output : Sorties de mod�lisation / nettoyage / caching 
  - dashboard : les  fichiers utilis�s par le backend du dashboard
- web : code pour g�n�rer le backend et le dashboard
  - app.py : backend Flask API
  - dashboard.py : Front end utilisant Streamlit


## Liens pour le dashboard : 
Le dashboard a �t� deploy� sur Heroku, pour y acc�der : 
http://pret-a-depenser.herokuapp.com/
le backend aussi a �t� deploy� sur Heroku, exemple d'appel : 
https://pret-a-depenser-backend.herokuapp.com/get_all_clients

