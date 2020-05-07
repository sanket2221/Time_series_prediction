from flask import request,render_template,Flask
from keras.models import load_model
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import date
scaler = MinMaxScaler(feature_range=(0, 1))

app = Flask(__name__)

model = load_model("model.h5")
'''
def data():

    return X_test'''


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict.py',methods=['POST'])
def predict():
    df = web.DataReader('BAJFINANCE.NS', data_source='yahoo', start='2012-01-01', end=date.today())
    model = load_model('model.h5')
    data = df.filter(['Close'])
    # Converting the dataframe to a numpy array
    dataset = data.values

    # Scale the all of the data to be values between 0 and 1

    scaled_data = scaler.fit_transform(dataset)

    last_60_days = data[-60:].values
    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # Create an empty list
    X_test = []
    # Append teh past 60 days
    X_test.append(last_60_days_scaled)
    # Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    return render_template('index.html', prediction='Closing Price Could be.{}'.format(pred_price))


if __name__ == '__main__':
    app.run(debug=True)

