#####################################################################
#                              TELCO
#####################################################################

# Telco is a fantastic California-based telecommunication company.
# They provided us with a dataset of 19 different feature of 7043 customers.
# Telco wants to be provided with a model which is able to predict the
# probability of a customer to be churn(leave the company).


# IMPORTING NECESSARY LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


# SETTING NECESSARY OPTIONS

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# IMPORTING DATASET

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()



df.columns = [col.upper() for col in df.columns]

def check_df(dataframe):
    print("##################### Head #####################")
    print(dataframe.head(10))
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("####################### NA ######################")
    print(dataframe.isnull().sum())
    print("################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)


df['TOTALCHARGES'].replace([' '], '0.0', inplace=True)
df["TOTALCHARGES"] = df["TOTALCHARGES"].astype(float)

df["CHURN"].replace(["Yes"], "1", inplace=True)
df["CHURN"].replace(["No"], "0", inplace=True)
df["CHURN"] = df["CHURN"].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, histogram=False, boxplot=False):
    quantiles = [0.05, 0.10, 0.50, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if histogram:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel("frequency")
        plt.title(numerical_col)
        plt.show()

    if boxplot:
        sns.boxplot(x=dataframe[numerical_col])
        plt.xlabel(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True, True)


###########################################################
#                  Feature Engineering
###########################################################

all_cat_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() > 1 and df[col].nunique() < 10]

triple_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() > 2 and df[col].nunique() < 10]

def label_encoder(dataframe, label_enc_col):
    labelencoder = LabelEncoder()
    dataframe[label_enc_col] = labelencoder.fit_transform(dataframe[label_enc_col])
    return dataframe

for col in all_cat_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, triple_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


dff = df.select_dtypes(include=['float', 'int'])
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff)
dff_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()
th = np.sort(dff_scores)[14]
clf_index = df[dff_scores < th].index
df.iloc[(clf_index)]
df.drop(index=clf_index, inplace=True)


df.loc[(df["MULTIPLELINES_1"] == 0) & (df["MULTIPLELINES_2"] == 0), "NEW_MUTLIPLELINES"] = 1
df.loc[(df["MULTIPLELINES_1"] == 0) & (df["MULTIPLELINES_2"] == 1), "NEW_MUTLIPLELINES"] = 0
df.loc[(df["MULTIPLELINES_1"] == 1) & (df["MULTIPLELINES_2"] == 0), "NEW_MUTLIPLELINES"] = 0

df.loc[(df["ONLINESECURITY_1"] == 0) & (df["ONLINESECURITY_2"] == 0), "NEW_ONLINESECURITY"] = 1
df.loc[(df["ONLINESECURITY_1"] == 0) & (df["ONLINESECURITY_2"] == 1), "NEW_ONLINESECURITY"] = 0
df.loc[(df["ONLINESECURITY_1"] == 1) & (df["ONLINESECURITY_2"] == 0), "NEW_ONLINESECURITY"] = 0

df.loc[(df["ONLINEBACKUP_1"] == 0) & (df["ONLINEBACKUP_2"] == 0), "NEW_ONLINEBACKUP"] = 1
df.loc[(df["ONLINEBACKUP_1"] == 0) & (df["ONLINEBACKUP_2"] == 1), "NEW_ONLINEBACKUP"] = 0
df.loc[(df["ONLINEBACKUP_1"] == 1) & (df["ONLINEBACKUP_2"] == 0), "NEW_ONLINEBACKUP"] = 0

df.loc[(df["DEVICEPROTECTION_1"] == 0) & (df["DEVICEPROTECTION_2"] == 0), "NEW_DEVICEPROTECTION"] = 1
df.loc[(df["DEVICEPROTECTION_1"] == 0) & (df["DEVICEPROTECTION_2"] == 1), "NEW_DEVICEPROTECTION"] = 0
df.loc[(df["DEVICEPROTECTION_1"] == 1) & (df["DEVICEPROTECTION_2"] == 0), "NEW_DEVICEPROTECTION"] = 0

df.loc[(df["TECHSUPPORT_1"] == 0) & (df["TECHSUPPORT_2"] == 0), "NEW_TECHSUPPORT"] = 1
df.loc[(df["TECHSUPPORT_1"] == 0) & (df["TECHSUPPORT_2"] == 1), "NEW_TECHSUPPORT"] = 0
df.loc[(df["TECHSUPPORT_1"] == 1) & (df["TECHSUPPORT_2"] == 0), "NEW_TECHSUPPORT"] = 0

df.loc[(df["STREAMINGTV_1"] == 0) & (df["STREAMINGTV_2"] == 0), "NEW_STREAMINGTV"] = 1
df.loc[(df["STREAMINGTV_1"] == 0) & (df["STREAMINGTV_2"] == 1), "NEW_STREAMINGTV"] = 0
df.loc[(df["STREAMINGTV_1"] == 1) & (df["STREAMINGTV_2"] == 0), "NEW_STREAMINGTV"] = 0

df.loc[(df["STREAMINGMOVIES_1"] == 0) & (df["STREAMINGMOVIES_2"] == 0), "NEW_STREAMINGMOVIES"] = 1
df.loc[(df["STREAMINGMOVIES_1"] == 0) & (df["STREAMINGMOVIES_2"] == 1), "NEW_STREAMINGMOVIES"] = 0
df.loc[(df["STREAMINGMOVIES_1"] == 1) & (df["STREAMINGMOVIES_2"] == 0), "NEW_STREAMINGMOVIES"] = 0


df["TOTAL_SERVICE"] = df["NEW_MUTLIPLELINES"] + df["NEW_ONLINESECURITY"] + df["NEW_ONLINEBACKUP"] \
                      + df["NEW_DEVICEPROTECTION"] + df["NEW_TECHSUPPORT"] \
                      + df["NEW_STREAMINGTV"] + df["NEW_STREAMINGMOVIES"]

list = df[["NEW_MUTLIPLELINES", "NEW_ONLINESECURITY", "NEW_ONLINEBACKUP", "NEW_DEVICEPROTECTION",
           "NEW_TECHSUPPORT", "NEW_STREAMINGTV", "NEW_STREAMINGMOVIES"]]

df = df.drop(list, axis=1)

df["TOTALCHARGES_PER_SERVICE"] = df["TOTALCHARGES"] / df["TOTAL_SERVICE"]

df[df["TOTAL_SERVICE"] == 0] = 1


cat_cols, num_cols, cat_but_car = grab_col_names(df)


rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])



from sklearn.ensemble import RandomForestClassifier

y = df["CHURN"]
X = df.drop(["CUSTOMERID", "CHURN"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
rf_model = RandomForestClassifier(random_state=44).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7994