import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

def lag(df,columns):
    for column in columns:
        #if column != 'mood' and column !='id':
        if column !='id':
            df[f'{column}__shift_b_{1}d'] = df[f'{column}'].shift(1)
            df[f'{column}__shift_b_{2}d'] = df[f'{column}'].shift(2)
            df[f'{column}__shift_b_{3}d'] = df[f'{column}'].shift(3)
            df[f'{column}__shift_b_{4}d'] = df[f'{column}'].shift(4)
            df[f'{column}__shift_b_{5}d'] = df[f'{column}'].shift(5)
    return df

def missing_value_(df,option=0):
    df = df.fillna(df.interpolate())
    df = df.dropna(subset=['mood'])

    df = df.fillna(0)
    return df

def normalization(df):
    for f in df.columns:
        if f == 'mood':
            continue
        fmin = df[f].min()
        fmax = df[f].max()
        df[f] = (df[f] - fmin)/(fmax-fmin)
    return  df

df = pd.read_csv('users_data.csv')

df_list = []
# get
patient_list = df.id.value_counts().keys()


for patient in patient_list:
    atrributes_list = ['mood', 'circumplex.valence', 'activity', 'circumplex.arousal', 'id']
    # get the data
    df_patient = df[df.id == patient].set_index('time')

    df_patient = missing_value_(df_patient)

    df_patient = df_patient[atrributes_list]

    df_patient = lag(df_patient, columns=atrributes_list)

    df_list.append(df_patient)

# concatenate the dataframes along axis 1
df_patients = pd.concat(df_list, axis=0)

# create a LabelEncoder object
le = LabelEncoder()
# fit the LabelEncoder object to the  id column and transform the data
df_patients['id_encoded'] = le.fit_transform(df_patients['id'])
# print the result
df_patients = df_patients.drop(columns=['id']).fillna(0)

# feature normalization
df_id = normalization(df_patients)

df_id['mood'] = round(df_id['mood'])

y = list(df_id['mood'])

scaled_data = df_id.drop(columns=['mood'])

X = scaled_data.to_numpy()
ACC_train = []
ACC_test = []
for solver in ['newton-cg','lbfgs','liblinear','sag']:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    logreg = LogisticRegression(solver=solver, random_state=1)

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    print('Acc', metrics.accuracy_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("r2:", metrics.r2_score(y_test, y_pred))

    y_pred_train = logreg.predict(X_train)

    ACC_test.append(metrics.accuracy_score(y_test, y_pred))
    ACC_train.append(metrics.accuracy_score(y_train, y_pred_train))

aa = ['newton-cg','lbfgs','liblinear','sag']
plt.plot(aa,ACC_test,label = 'test acc')
plt.plot(aa,ACC_train,label = 'train acc')
plt.ylabel('acc', size=15)
plt.xlabel('solver', size=15)
plt.legend(fontsize=15)
#plt.show()
plt.savefig('lg para.pdf',dpi = 200,bbox_inches = 'tight')
