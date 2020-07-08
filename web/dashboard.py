import json
import pickle
from urllib.request import urlopen

import altair as alt
import lightgbm as lgb
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt

END_POINT = 'http://127.0.0.1:5000/'


@st.cache
def load_main_data():
    """

    :return: the sample data frame from backend
    """
    json_url = urlopen(END_POINT + 'get_clients_df')
    data = json.loads(json_url.read())
    return pd.read_json(data)


@st.cache
def load_all_clients():
    """

    :return: all clients ids from backend
    """
    json_url = urlopen(END_POINT + 'get_all_clients')
    return json.loads(json_url.read())


@st.cache
def load_stats():
    """

    :return: stats data from backend
    """
    json_url = urlopen(END_POINT + 'get_stats')
    data = json.loads(json_url.read())
    return pd.read_json(data)


@st.cache
def load_model():
    """

    :return: loading the model
    """
    return pickle.load(open('../output/best_estimator.pkl', 'rb'))


@st.cache
def load_shap_explainer():
    """

    :return: a built shap explainer
    """
    data = load_main_data()
    data = data.drop(columns=['PREDICT', 'PREDICT_PROBA', 'SK_ID_CURR'])
    explainer = shap.TreeExplainer(load_model())
    shap_values = explainer.shap_values(data)
    return explainer, shap_values


def get_lime_data(id):
    """

    :param id: the client id
    :return: the lime explanation for top features
    """
    json_url = urlopen(END_POINT + 'get_lime/' + str(id))
    return pd.read_json(json.loads(json_url.read()))


def get_client_prediction(id):
    """

    :param id: client id
    :return: prediction / prediction proba for the client
    """
    json_url = urlopen(END_POINT + 'get_client_prediction/' + str(id))
    return json.loads(json_url.read())


def get_knn_data(id):
    """

    :param id: the client id
    :return: the nearest neighbors avergage on different features
    """
    json_url = urlopen(END_POINT + 'get_knn/' + str(id))
    return pd.read_json(json.loads(json_url.read()))


def build_stats_data_frame(id, features, data):
    """

    :param id: the client ids
    :param features: the columns on which display stats (lime data)
    :param data: the whole dataframe
    :return: a stats dataframe built to display a chart
    """
    df = pd.DataFrame(columns=['Feature', 'Type', 'Mean'])
    knn_df = get_knn_data(id)
    stats_df = load_stats()
    client_data = data[data['SK_ID_CURR'] == id]
    i = 0
    predict = get_client_prediction(id)['prediction']
    stats_for_same_profile = stats_df[stats_df['TYPE'] == str(predict)]
    stats_for_all = stats_df[stats_df['TYPE'] == 'ALL']
    for feature in features:
        df.loc[i] = [feature, 'Client', client_data[feature].iloc[0]]
        i += 1
        df.loc[i] = [feature, 'Near clients', knn_df[feature].iloc[0]]
        i += 1
        df.loc[i] = [feature, 'Same clients', stats_for_same_profile[feature].iloc[0]]
        i += 1
        df.loc[i] = [feature, 'All clients', stats_for_all[feature].iloc[0]]
        i += 1

    return df


def main():
    # add a title
    st.title('Prêt à dépenser : Implémenter un modèle de scoring')
    st.subheader('By Salaheddine E.G : Openclassroom student')

    # data preview
    data = load_main_data()
    st.subheader('Apercu des données')
    st.write('Aperçu des données pour les gens éligibles à un crédit : ')
    st.write(data.head(5))
    st.write('...Et pour ceux pas éligibles : ')
    st.write(data.tail(5))

    # get data from api to choose a client
    all_clients = load_all_clients()
    st.subheader('Affichage des données pour un client')

    id = st.selectbox('Choisissez un client : ', all_clients)
    # display the prediction proba
    predict = get_client_prediction(id)
    st.info('Probabilité de défaut du client %.2f %%' % (predict['prediction_proba'] * 100))

    # Display lime explainer using altair chart
    st.subheader('Explication lime du résultat pour le client')
    lime_data = get_lime_data(id)
    lime_data.dropna(how='all', axis=1, inplace=True)
    lime_draw = lime_data.T
    lime_draw.rename(columns={lime_draw.columns[0]: 'value'}, inplace=True)
    lime_draw['feature'] = lime_draw.index

    st.write(alt.Chart(lime_draw).mark_bar().encode(
        x='value',
        y='feature',
        color=alt.condition(
            alt.datum.value > 0,
            alt.value('coral'),  # The positive color
            alt.value('green')  # The negative color
        )
    ).properties(width=600, height=400))

    # display clients data vs similar clients, clients with the same target value, and vs all clients
    st.subheader('Comparaison du résultat de clients par rapport à d\'autres clients')
    stats = build_stats_data_frame(id, lime_data.columns, data)

    first_features = lime_data.columns[0:5]
    last_features = lime_data.columns[5:]

    st.write(alt.Chart(stats[stats['Feature'].isin(first_features)]).mark_bar().encode(
        x='Type:N',
        y='Mean:Q',
        color='Type:N',
        column='Feature:N'
    ).properties(width=200, height=400))

    st.write(alt.Chart(stats[stats['Feature'].isin(last_features)]).mark_bar().encode(
        x='Type:N',
        y='Mean:Q',
        color='Type:N',
        column='Feature:N'
    ).properties(width=200, height=400))

    # display lgb model importance
    st.subheader('Explication du modèle ')
    model = load_model()
    lgb.plot_importance(model, max_num_features=10)
    plt.title('Light GBM importance')
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # display shap explanation
    st.subheader('Explication du modèle en utilisant SHAP')
    explainer, shap_values = load_shap_explainer()
    plt.title('Importance des features en utilisant shap')
    shap_data = data.drop(columns=['PREDICT', 'PREDICT_PROBA', 'SK_ID_CURR'])
    shap.summary_plot(shap_values, shap_data, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()
