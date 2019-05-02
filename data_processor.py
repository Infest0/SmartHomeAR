import numpy as np
import pandas as pd
import itertools
import os.path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class Processor:
    def __init__(self, sliding_window_length=6,
                 columns_names=('date', 'time', 'sensor_id', 'value', 'activity1', 'activity2', 'activity3', 'activity4',
                                'activity5'),
                 data_path=None, date='2011-01-21', date_sign='less', sensor_encoder=None, activity_encoder=None,
                 mutual_matrix=None, sens_num=None, acts_num=None):
        self.columns_names = columns_names
        self.sliding_window_length = sliding_window_length
        self.data_path = data_path
        self.date = date
        self.date_sign = date_sign
        self.sensor_encoder = sensor_encoder
        self.activity_encoder = activity_encoder
        self.mutual_matrix = mutual_matrix
        self.sens_num = sens_num
        self.acts_num = acts_num
        self.data = None
        self.processed = None
        self.data = pd.read_csv(data_path, header=None, sep='\s+', names=self.columns_names)

    def markdown_acitivities(self):
        activity_pool = []

        for index, row in self.data.iterrows():
            if row.activity2 == 'begin':
                break
            else:
                self.data.drop(index, inplace=True)

        for index, row in self.data.iterrows():
            if row.activity2 == 'begin':
                activity_pool.append(row.activity1)
            elif row.activity2 == 'end':
                activity_pool.remove(row.activity1)
            elif activity_pool:
                if len(activity_pool) > 1:
                    self.data.drop(index, inplace=True)
                else:
                    self.data.at[index, 'activity1'] = activity_pool[-1]

    def calculate_mutual_info_matrix(self, mutual_matrix, sensnum):
        current_activity = self.data.iloc[0].activity1
        wind = []
        winds=[]
        winnum = 0

        for idx, row in self.data.iterrows():
            if current_activity == row.activity1:
                wind.append(row.sensor_id)
            else:
                sensors = np.unique(wind)
                for sensor in range(0, sensnum):
                    for sensor2 in range(0, sensnum):
                        if sensor in sensors and sensor2 in sensors:
                            mutual_matrix[sensor][sensor2] += 1
                            winds.append(wind)
                winnum += 1
                current_activity = row.activity1
                wind = [row.sensor_id]

        return mutual_matrix/winnum

    def process(self):
        self.data['datetime'] = pd.to_datetime(self.data['date'] + " " + self.data['time'])

        if self.date_sign == 'more':
            self.data = self.data[(self.data['datetime'] > self.date)]
        else:
            self.data = self.data[self.data['datetime'] < self.date]

        vls = ['begin', 'end']

        #There are datasets that have activity name in
        for index, row in self.data.iterrows():
            if not row.activity2 in vls and not row.activity2 is np.nan:
                if not row.activity3 in vls:
                    if not row.activity4 in vls:
                        self.data.loc[index, 'activity1'] = row.activity1 + "_" + row.activity2 + "_" + row.activity3 + "_" + row.activity4
                        self.data.loc[index, 'activity2'] = row.activity5
                    else:
                        self.data.loc[index, 'activity1'] = row.activity1 + "_" + row.activity2 + "_" + row.activity3
                        self.data.loc[index, 'activity2'] = row.activity4
                else:
                    self.data.loc[index, 'activity1'] = row.activity1 + "_" + row.activity2
                    self.data.loc[index, 'activity2'] = row.activity3

        #There are some typos in dataset in status column
        one_vals = ['ON', 'OPEN', 'ONc', 'ON5', 'ON55', 'ON5c', 'ONcc', 'ONc5c', 'ONc5', 'OPENc', 'O', 'ONM026',
                    'ONM009', 'ONM024']
        zero_vals = ['OFF', 'CLOSE', 'OFF5', 'OFcF', 'OFFc', 'OFFcc', 'OFF5cc', 'OFF5c', 'OFFc5', 'OcFF', 'OFFccc5',
                     'OF', 'CLOSED']

        self.data.loc[self.data['value'].isin(one_vals), 'value'] = 1
        self.data.loc[self.data['value'].isin(zero_vals), 'value'] = 0
        self.markdown_acitivities()
        #we need no data from temperature data
        self.data = self.data[~self.data.sensor_id.str.contains('c')]
        self.data = self.data[~self.data.sensor_id.str.startswith('T')]

        self.data.loc[self.data['activity1'].isnull(), 'activity1'] = 'other'
        self.data.drop(columns=['date', 'time'], inplace=True)

        if self.sensor_encoder is None:
            self.sensor_encoder = LabelEncoder()
            self.sensor_encoder.fit(self.data.sensor_id)

        if self.activity_encoder is None:
            self.activity_encoder = LabelEncoder()
            self.activity_encoder.fit(self.data.activity1.astype(str))

        # We are not dealing with other activity here
        self.data = self.data[self.data.activity1 != 'other']

        self.data.sensor_id = self.sensor_encoder.transform(self.data.sensor_id)
        self.data.activity1 = self.activity_encoder.transform(self.data.activity1.astype(str))

        if self.sens_num is None:
            self.sens_num = len(self.data.sensor_id.unique())

        if self.acts_num is None:
            self.acts_num = len(self.data.activity1.unique())

        self.data = self.data.reset_index()

        if self.mutual_matrix is None:
            self.mutual_matrix = self.calculate_mutual_info_matrix(np.zeros((self.sens_num, self.sens_num)),
                                                                   self.sens_num)

    def get_segments(self):
        windows_vectors, activities_vectors = [], [],

        for idx in range(0, self.data.shape[0], 1):
            current_sens = []
            if self.data.shape[0] <= idx + self.sliding_window_length:
                break

            freq = np.zeros(self.sens_num)
            windows = self.data.iloc[idx:idx + self.sliding_window_length]
            last_activity = windows.iloc[self.sliding_window_length - 1].activity1

            start_time = windows.iloc[0].datetime
            last_time = windows.iloc[self.sliding_window_length - 1].datetime
            last_time = datetime(last_time.year, last_time.month, last_time.day, last_time.hour, last_time.minute,
                                 last_time.second)
            start_time = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute,
                                  start_time.second)
            last_time = last_time.hour / 24  + last_time.minute / 60. + last_time.second / 3600.
            start_time = start_time.hour / 24 + start_time.minute / 60. + start_time.second / 3600.
            last_sensor_id = windows.iloc[self.sliding_window_length - 1].sensor_id
            second_last_id = windows.iloc[self.sliding_window_length - 2].sensor_id

            for ind, act in windows.iterrows():
                freq[int(act.sensor_id)] += 1
            
            #weighting frequencies using mutual matrix table
            for ind, val in enumerate(freq):
                freq[ind] = freq[ind] * self.mutual_matrix[last_sensor_id][ind]

            timespan = last_time - start_time
            freq = np.append(freq, (last_time, timespan, second_last_id, last_sensor_id))

            windows_vectors.append(freq)
            activities_vectors.append(last_activity)

        return np.array(windows_vectors), np.array(activities_vectors)


def get_train_test_data(path, date1, date2):
    train_processor = Processor(date_sign='less', data_path=path, date=date1)
    train_processor.process()
    test_processor = Processor(date_sign='more', data_path=path, date=date2,
                               activity_encoder=train_processor.activity_encoder,
                               sensor_encoder=train_processor.sensor_encoder,
                               mutual_matrix=train_processor.mutual_matrix, sens_num=train_processor.sens_num,
                               acts_num=train_processor.acts_num)
    test_processor.process()
    train_x, train_y = train_processor.get_segments()
    test_x, test_y = test_processor.get_segments()
    return train_x, train_y, test_x, test_y, train_processor.activity_encoder.classes_


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()