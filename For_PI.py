import tensorflow as tf
# import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# data = {
#     'a1': [-3.93468,
#             2.40285,
#             6.24273,
#             -3.99411,
#             1.76551
#             ],
#  'a2': [6.55216,
#   9.99438,
#   -3.17577,
#   -14.1448,
#   -6.46492],
#  'a3' : [-1.23798,
#   -3.24265,
#   -0.686974,
#   3.84536,
#   5.09623],
#  'a4':[20.3103,
#   8.3132,
#   -4.19382,
#   -5.6586,
#   -9.00136],
#  'load':[50, 50, 50, 50, 50]
#  }

data = {
  "a1": [-3.93468,2.40285,6.24273,-3.99411,1.76551,8.0413,-4.01812,4.91148,9.87637,-4.45857,-5.11952,-9.42848,-3.65882,0.166275,0.325032
  ],
  "a2": [
    6.55216,9.99438,-3.17577,-14.1448,-6.46492,0.962914,1.07599,-4.4753,1.05646,5.19339,5.6831,5.67469,1.68175,-0.326172,5.88362
  ],
  "a3": [-1.23798,-3.24265,-0.686974,3.84536,5.09623,-1.54231,7.16725,4.65973,-6.2591,-13.3042,-10.0638,0.432831,7.39364,2.70997,-0.036916
  ],
  "a4": [20.3103,8.3132,-4.19382,-5.6586,-9.00136,2.54418,5.76429,1.98554,0.918555,-2.54888,0.417422,-2.98862,-5.52874,7.74519,10.6415
  ],

  "load": [50,50,50,50,50,50,50,50,50,50,50,50,50,50,50
  ]
}
# # Healthy
data = {
  "a1":[-10.8023,13.9087,6.37438,-15.2817,3.13954,0.677448,-10.5754,-4.03329,1.86867,7.58148],
 "a2":[-5.15393,7.88091,-3.59342,3.73856,6.8219,-3.23441,7.7254,2.57692,-5.0894,6.20596],
 "a3":[6.84189,-2.98886,0.590603,4.00437,-2.92981,-1.72599,-2.18401,1.46843,5.34229,-6.12133],
 "a4":[-3.60985,6.41759,2.61505,1.52184,1.55903,-3.14302,2.56965,2.72891,-1.36563,11.5483],
 "load": [90,90,90,90,90,90,90,90,90,90]
}

# data is for broken 
# Convert to JSON to dataframe
df = pd.DataFrame.from_dict(data, orient='index').T
			
sensor_readings = df.melt(
    id_vars=['load'],
    value_vars=['a1','a2','a3','a4'],
    var_name='sensor',
    value_name='reading'
)
sensor_readings
sensor_readings.shape

data = []
for (load,sensor),g in sensor_readings.groupby(['load','sensor']):
    # print("load: {0} for sensor: {1}".format(load,sensor))
    # print("g.reading.values:",g.reading.values)
    vals = g.reading.values
    splits = np.split(vals, range(1000,vals.shape[0],1000))
    # print(load,sensor, splits[:])
    # print("splits.shape:",len(splits[0]))
    for i,s in enumerate(splits):  # except the last one
        # print("i:",i)
        data.append({
            'sensor_a1': int(sensor=='a1'),
            'sensor_a2': int(sensor=='a2'),
            'sensor_a3': int(sensor=='a3'),
            # no need to put a4: if a1-3 are 0, then it's sensor a4
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

model = tf.keras.models.load_model('mdoels\model_700_r.h5')
prediction = model.predict(df_data.values)

y_label = {0:"Healthy",1:"Broken"}
print(y_label[np.max(np.round(prediction[0]))])



