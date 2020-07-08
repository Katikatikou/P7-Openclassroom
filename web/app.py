import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

print('Reading knn dataframe ...')
knn_df = pd.read_feather('../output/dashboard/knn.ft')
print('Reading Lime explanation dataframe ... ')
lime_df = pd.read_feather('../output/dashboard/lime.ft')
print('Reading stats dataframe ... ')
stats_df = pd.read_feather('../output/dashboard/stats.ft')
print('Reading sample dataframe ...')
data = pd.read_feather('../output/dashboard/sample.ft')
print('Backend is ready dude!')


@app.route('/get_all_clients', methods=['GET'])
def get_all_clients():
    """

    :return: all clients ids in the sample
    """
    return jsonify(list(data['SK_ID_CURR']))


@app.route('/get_client_prediction/<id>', methods=['GET'])
def get_client_prediction(id):
    """

    :param id: the client id
    :return: a dict with the client prediction, and the proba to be at the default
    """
    X = data[data['SK_ID_CURR'] == int(id)]
    result = {}
    if X.shape[0] == 0:
        return result
    result['prediction'] = list(X['PREDICT'])[0]
    result['prediction_proba'] = list(X['PREDICT_PROBA'])[0]
    return jsonify(result)


@app.route('/get_clients_df')
def get_clients_df():
    """

    :return: the whole sample data
    """
    return jsonify(data.to_json())


@app.route('/get_lime/<id>')
def get_lime(id):
    """

    :param id: the client id
    :return: lime explanation data for this client
    """
    X = lime_df[lime_df['SK_ID_CURR'] == int(id)]
    if X.shape[0] == 0:
        return {}
    return jsonify(X.drop(columns='SK_ID_CURR').to_json())


@app.route('/get_knn/<id>')
def get_knn(id):
    """

    :param id: the client id
    :return: nearest neighbors average on different features
    """
    X = knn_df[knn_df['SK_ID_CURR'] == int(id)]
    if X.shape[0] == 0:
        return {}
    return jsonify(X.to_json())


@app.route('/get_stats')
def get_stats():
    """

    :return: stats data containing mean per feature for different features (per good / bad clients and all clients)
    """
    return jsonify(stats_df.to_json())


if __name__ == '__main__':
    app.run()
