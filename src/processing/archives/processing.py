# base processing code
import pandas as pd
import numpy as np
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import category_encoders as ce

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from tqdm import tqdm

# ingest data 
def ingest(data_pth):
    data = pd.read_csv(data_pth, sep=';')
    return data

# explore categorical features
def plot_bar(column):
    # temp df 
    temp_1 = pd.DataFrame()
    # count categorical values
    temp_1['No_deposit'] = data[data['y'] == 'no'][column].value_counts()
    temp_1['Yes_deposit'] = data[data['y'] == 'yes'][column].value_counts()
    temp_1.plot(kind='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Number of clients')
    plt.title('Distribution of {} and deposit'.format(column))
    plt.show();
    plt.savefig.plot_bar('job'), plot_bar('marital'), plot_bar('education'), plot_bar('contact'), plot_bar('loan'), plot_bar('housing')

# build correlation matrix
def correlation_matrix(data):
    corr = data.corr()
    plt.savefig.corr.style.background_gradient(cmap='PuBu')

def feature_engineering(data):
    # Replacing values with binary ()
    data.contact = data.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8') 
    data.loan = data.loan.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
    data.housing = data.housing.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
    data.default = data.default.map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')
    data.pdays = data.pdays.replace(999, 0) # replace with 0 if not contact 
    data.previous = data.previous.apply(lambda x: 1 if x > 0 else 0).astype('uint8') # binary has contact or not

    # binary if were was an outcome of marketing campane
    data.poutcome = data.poutcome.map({'nonexistent':0, 'failure':0, 'success':1}).astype('uint8') 

    # change the range of Var Rate
    data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: x*-0.0001 if x > 0 else x*1)
    data['emp.var.rate'] = data['emp.var.rate'] * -1
    data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: -np.log(x) if x < 1 else np.log(x)).astype('uint8')

    # Multiply consumer index 
    data['cons.price.idx'] = (data['cons.price.idx'] * 10).astype('uint8')

    # change the sign (we want all be positive values)
    data['cons.conf.idx'] = data['cons.conf.idx'] * -1

    # re-scale variables
    data['nr.employed'] = np.log2(data['nr.employed']).astype('uint8')
    data['cons.price.idx'] = np.log2(data['cons.price.idx']).astype('uint8')
    data['cons.conf.idx'] = np.log2(data['cons.conf.idx']).astype('uint8')
    data.age = np.log(data.age)

    # less space
    data.euribor3m = data.euribor3m.astype('uint8')
    data.campaign = data.campaign.astype('uint8')
    data.pdays = data.pdays.astype('uint8')

# fucntion to One Hot Encoding
def encode(data, col):
    return pd.concat([data, pd.get_dummies(col, prefix=col.name)], axis=1)

def one_hot_encoding(data):
    # One Hot encoding of 3 variable 
    data = encode(data, data.job)
    data = encode(data, data.month)
    data = encode(data, data.day_of_week)

    # Drop tranfromed features
    data.drop(['job', 'month', 'day_of_week'], axis=1, inplace=True)


def convert_duration(data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration'] = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration'] = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 645), 'duration'] = 4
    data.loc[data['duration']  > 645, 'duration'] = 5
    return data
# duration(data);