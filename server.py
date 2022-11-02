from flask import Flask, request, jsonify
import tensorflow as tf
# import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

model = tf.keras.models.load_model('./mdoels/model_700_r.h5')


@app.route('/')
def home():
    return 'Welcome to the home page'


def clear_data(sensor_readings):
    data = []

    for (load, sensor), g in sensor_readings.groupby(['load', 'sensor']):
        vals = g.reading.values
        splits = np.split(vals, range(1000, vals.shape[0], 1000))
        for i, s in enumerate(splits):
            data.append({
                'sensor_a1': int(sensor == 'a1'),
                'sensor_a2': int(sensor == 'a2'),
                'sensor_a3': int(sensor == 'a3'),
                'load': load,
                'mean': np.mean(s),
                'std': np.std(s),
                'kurt': stats.kurtosis(s),
                'skew': stats.skew(s),
                'moment': stats.moment(s),
            })

    df_data = pd.DataFrame(data)
    # print(df_data)
    data = df_data.values
    cols_to_scale = ['load', 'mean', 'std', 'kurt', 'skew', 'moment']
    scaler = MinMaxScaler()
    df_data[cols_to_scale] = scaler.fit_transform(df_data[cols_to_scale])

    return df_data


@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the POST request.
    data = request.get_json(force=True)
    df = pd.DataFrame.from_dict(data, orient='index').T

    sensor_readings = df.melt(
        id_vars=['load'],
        value_vars=['a1', 'a2', 'a3', 'a4'],
        var_name='sensor',
        value_name='reading'
    )

    df_data = clear_data(sensor_readings)
    # make prediction using model loaded from disk as per the data.
    prediction = model.predict([np.array(list(df_data.values()))])

    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    print('Starting Python Flask Server For Load Prediction')
    app.run(port=5000, debug=True, host='0.0.0.0')
