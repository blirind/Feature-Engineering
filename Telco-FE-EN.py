#####################################################################
#                              TELCO
#####################################################################

# Telco is a fantastic California-based telecommunication company.
# They provided us with a dataset of 19 different feature of 7043 customers.
# Telco wants to be provided with a model which is able to predict the
# probability of a customer to be churn(leave the company).


# PLEASE FIND THE WHOLE FUNCTION IN THE END OR GO STEP BY STEP STARTING HERE:


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
from sklearn.ensemble import RandomForestClassifier

# SETTING NECESSARY OPTIONS

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# IMPORTING DATASET

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

# PLEASE FIND THE WHOLE FUNCTION IN THE END OR GO STEP BY STEP STARTING HERE:


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

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


dff = df.select_dtypes(include=['float', 'int'])
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff)
dff_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()
th = np.sort(dff_scores)[11]
clf_index = df[dff_scores < th].index
df.iloc[(clf_index)]
df.drop(index=clf_index, inplace=True)


df.loc[(df["MULTIPLELINES_No phone service"] == 0) & (df["MULTIPLELINES_Yes"] == 0), "NEW_MUTLIPLELINES"] = 0
df.loc[(df["MULTIPLELINES_No phone service"] == 0) & (df["MULTIPLELINES_Yes"] == 1), "NEW_MUTLIPLELINES"] = 1
df.loc[(df["MULTIPLELINES_No phone service"] == 1) & (df["MULTIPLELINES_Yes"] == 0), "NEW_MUTLIPLELINES"] = 0

df.loc[(df["ONLINESECURITY_No internet service"] == 0) & (df["ONLINESECURITY_Yes"] == 0), "NEW_ONLINESECURITY"] = 0
df.loc[(df["ONLINESECURITY_No internet service"] == 0) & (df["ONLINESECURITY_Yes"] == 1), "NEW_ONLINESECURITY"] = 1
df.loc[(df["ONLINESECURITY_No internet service"] == 1) & (df["ONLINESECURITY_Yes"] == 0), "NEW_ONLINESECURITY"] = 0

df.loc[(df["ONLINEBACKUP_No internet service"] == 0) & (df["ONLINEBACKUP_Yes"] == 0), "NEW_ONLINEBACKUP"] = 0
df.loc[(df["ONLINEBACKUP_No internet service"] == 0) & (df["ONLINEBACKUP_Yes"] == 1), "NEW_ONLINEBACKUP"] = 1
df.loc[(df["ONLINEBACKUP_No internet service"] == 1) & (df["ONLINEBACKUP_Yes"] == 0), "NEW_ONLINEBACKUP"] = 0

df.loc[(df["DEVICEPROTECTION_No internet service"] == 0) & (df["DEVICEPROTECTION_Yes"] == 0), "NEW_DEVICEPROTECTION"] = 0
df.loc[(df["DEVICEPROTECTION_No internet service"] == 0) & (df["DEVICEPROTECTION_Yes"] == 1), "NEW_DEVICEPROTECTION"] = 1
df.loc[(df["DEVICEPROTECTION_No internet service"] == 1) & (df["DEVICEPROTECTION_Yes"] == 0), "NEW_DEVICEPROTECTION"] = 0

df.loc[(df["TECHSUPPORT_No internet service"] == 0) & (df["TECHSUPPORT_Yes"] == 0), "NEW_TECHSUPPORT"] = 0
df.loc[(df["TECHSUPPORT_No internet service"] == 0) & (df["TECHSUPPORT_Yes"] == 1), "NEW_TECHSUPPORT"] = 1
df.loc[(df["TECHSUPPORT_No internet service"] == 1) & (df["TECHSUPPORT_Yes"] == 0), "NEW_TECHSUPPORT"] = 0

df.loc[(df["STREAMINGTV_No internet service"] == 0) & (df["STREAMINGTV_Yes"] == 0), "NEW_STREAMINGTV"] = 0
df.loc[(df["STREAMINGTV_No internet service"] == 0) & (df["STREAMINGTV_Yes"] == 1), "NEW_STREAMINGTV"] = 1
df.loc[(df["STREAMINGTV_No internet service"] == 1) & (df["STREAMINGTV_Yes"] == 0), "NEW_STREAMINGTV"] = 0

df.loc[(df["STREAMINGMOVIES_No internet service"] == 0) & (df["STREAMINGMOVIES_Yes"] == 0), "NEW_STREAMINGMOVIES"] = 0
df.loc[(df["STREAMINGMOVIES_No internet service"] == 0) & (df["STREAMINGMOVIES_Yes"] == 1), "NEW_STREAMINGMOVIES"] = 1
df.loc[(df["STREAMINGMOVIES_No internet service"] == 1) & (df["STREAMINGMOVIES_Yes"] == 0), "NEW_STREAMINGMOVIES"] = 0

# TOTAL HİZMET SAYISI:
df["TOTAL_SERVICE"] = df["NEW_MUTLIPLELINES"] + df["NEW_ONLINESECURITY"] + df["NEW_ONLINEBACKUP"] \
                      + df["NEW_DEVICEPROTECTION"] + df["NEW_TECHSUPPORT"] \
                      + df["NEW_STREAMINGTV"] + df["NEW_STREAMINGMOVIES"] + df["PHONESERVICE_Yes"]

list = df[["NEW_MUTLIPLELINES", "NEW_ONLINESECURITY", "NEW_ONLINEBACKUP", "NEW_DEVICEPROTECTION",
           "NEW_TECHSUPPORT", "NEW_STREAMINGTV", "NEW_STREAMINGMOVIES"]]

df = df.drop(list, axis=1)


df["TOTALCHARGES_PER_SERVICE"] = df["TOTALCHARGES"] / df["TOTAL_SERVICE"]

df["TOTALCHARGES_PER_SERVICE"].replace(np.inf, 0, inplace=True)


low_mean = df[(df["TOTALCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 0)]["TOTALCHARGES"].mean()
high_mean = df[(df["TOTALCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 1)]["TOTALCHARGES"].mean()

df.loc[(df["TOTALCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 0), "TOTALCHARGES_PER_SERVICE"] = low_mean
df.loc[(df["TOTALCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 1), "TOTALCHARGES_PER_SERVICE"] = high_mean


df["MONTHLYCHARGES_PER_SERVICE"] = df["MONTHLYCHARGES"] / df["TOTAL_SERVICE"]
df["MONTHLYCHARGES_PER_SERVICE"].replace(np.inf, 0, inplace=True)

low_mean = df[(df["MONTHLYCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 0)]["MONTHLYCHARGES"].mean()
high_mean = df[(df["MONTHLYCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 1)]["MONTHLYCHARGES"].mean()

df.loc[(df["MONTHLYCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 0), "MONTHLYCHARGES_PER_SERVICE"] = low_mean
df.loc[(df["MONTHLYCHARGES_PER_SERVICE"] == 0) & (df["PARTNER_Yes"] == 1), "MONTHLYCHARGES_PER_SERVICE"] = high_mean


zero_mean = df.loc[(df["CONTRACT_One year"] == 0) & (df["CONTRACT_Two year"] == 0)]["TOTALCHARGES"].mean()
two_mean = df.loc[(df["CONTRACT_One year"] == 0) & (df["CONTRACT_Two year"] == 1)]["TOTALCHARGES"].mean()
one_mean = df.loc[(df["CONTRACT_One year"] == 1) & (df["CONTRACT_Two year"] == 0)]["TOTALCHARGES"].mean()


df.loc[(df["CONTRACT_One year"] == 0) & (df["CONTRACT_Two year"] == 0), "CONTRACT_TYPE"] = 1
df.loc[(df["CONTRACT_One year"] == 0) & (df["CONTRACT_Two year"] == 1), "CONTRACT_TYPE"] = two_mean
df.loc[(df["CONTRACT_One year"] == 1) & (df["CONTRACT_Two year"] == 0), "CONTRACT_TYPE"] = one_mean


cat_cols, num_cols, cat_but_car = grab_col_names(df)


rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])



y = df["CHURN"]
X = df.drop(["CUSTOMERID", "CHURN"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
rf_model = RandomForestClassifier(random_state=18).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.8054

############################################################################################
#                                       F U N C T I O N
############################################################################################




import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()


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

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

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

def target_analyser(dataframe, target, num_cols, cat_cols):
    for col in dataframe.columns:
        if col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        if col in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[col].mean()}), end="\n\n\n")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def label_encoder(dataframe, label_enc_col):
    labelencoder = LabelEncoder()
    dataframe[label_enc_col] = labelencoder.fit_transform(dataframe[label_enc_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")






def telco_model(data, target, check_df_df=False, cat_summary_df=False,
                col_summary_df=False, check_outlier_df=False, correlation=False,
                LOF=False, new_features=False, std_scale=False, rob_scale=True):

    data.columns = [col.upper() for col in data.columns]

    if check_df_df:
        check_df(data)

    data['TOTALCHARGES'].replace([' '], '0.0', inplace=True)
    data["TOTALCHARGES"] = data["TOTALCHARGES"].astype(float)
    data["CHURN"].replace(["Yes"], "1", inplace=True)
    data["CHURN"].replace(["No"], "0", inplace=True)
    data["CHURN"] = data["CHURN"].astype(int)

    cat_cols, num_cols, cat_but_car = grab_col_names(data, 10, 20)

    if cat_summary_df:
        for i in cat_cols:
            cat_summary(data, i, True)

    if col_summary_df:
        for i in num_cols:
            num_summary(data, i, True, True)

    if check_outlier_df:
        for i in num_cols:
            s = check_outlier(data, num_cols)
            print(f"Is there any outlier value for {i}?: {s}")

    target_analyser(data, target, num_cols, cat_cols)

    data.isnull().any().any()

    if correlation:
        corr = data.corr()
        sns.heatmap(corr, annot=True)
        plt.show()

    data = one_hot_encoder(data, cat_cols)

    cat_cols, num_cols, cat_but_car = grab_col_names(data, 10, 20)

    if LOF:
        dff = data.select_dtypes(include=['float', 'int'])
        clf = LocalOutlierFactor(n_neighbors=20)
        clf.fit_predict(dff)
        dff_scores = clf.negative_outlier_factor_
        th = np.sort(dff_scores)[11]
        clf_index = df[dff_scores < th].index
        data.drop(index=clf_index, inplace=True)

    if new_features:
        data.loc[(data["MULTIPLELINES_No phone service"] == 0) & (data["MULTIPLELINES_Yes"] == 0), "NEW_MUTLIPLELINES"] = 0
        data.loc[(data["MULTIPLELINES_No phone service"] == 0) & (data["MULTIPLELINES_Yes"] == 1), "NEW_MUTLIPLELINES"] = 1
        data.loc[(data["MULTIPLELINES_No phone service"] == 1) & (data["MULTIPLELINES_Yes"] == 0), "NEW_MUTLIPLELINES"] = 0

        data.loc[(data["ONLINESECURITY_No internet service"] == 0) & (data["ONLINESECURITY_Yes"] == 0), "NEW_ONLINESECURITY"] = 0
        data.loc[(data["ONLINESECURITY_No internet service"] == 0) & (data["ONLINESECURITY_Yes"] == 1), "NEW_ONLINESECURITY"] = 1
        data.loc[(data["ONLINESECURITY_No internet service"] == 1) & (data["ONLINESECURITY_Yes"] == 0), "NEW_ONLINESECURITY"] = 0

        data.loc[(data["ONLINEBACKUP_No internet service"] == 0) & (data["ONLINEBACKUP_Yes"] == 0), "NEW_ONLINEBACKUP"] = 0
        data.loc[(data["ONLINEBACKUP_No internet service"] == 0) & (data["ONLINEBACKUP_Yes"] == 1), "NEW_ONLINEBACKUP"] = 1
        data.loc[(data["ONLINEBACKUP_No internet service"] == 1) & (data["ONLINEBACKUP_Yes"] == 0), "NEW_ONLINEBACKUP"] = 0

        data.loc[(data["DEVICEPROTECTION_No internet service"] == 0) & (data["DEVICEPROTECTION_Yes"] == 0), "NEW_DEVICEPROTECTION"] = 0
        data.loc[(data["DEVICEPROTECTION_No internet service"] == 0) & (data["DEVICEPROTECTION_Yes"] == 1), "NEW_DEVICEPROTECTION"] = 1
        data.loc[(data["DEVICEPROTECTION_No internet service"] == 1) & (data["DEVICEPROTECTION_Yes"] == 0), "NEW_DEVICEPROTECTION"] = 0

        data.loc[(data["TECHSUPPORT_No internet service"] == 0) & (data["TECHSUPPORT_Yes"] == 0), "NEW_TECHSUPPORT"] = 0
        data.loc[(data["TECHSUPPORT_No internet service"] == 0) & (data["TECHSUPPORT_Yes"] == 1), "NEW_TECHSUPPORT"] = 1
        data.loc[(data["TECHSUPPORT_No internet service"] == 1) & (data["TECHSUPPORT_Yes"] == 0), "NEW_TECHSUPPORT"] = 0

        data.loc[(data["STREAMINGTV_No internet service"] == 0) & (data["STREAMINGTV_Yes"] == 0), "NEW_STREAMINGTV"] = 0
        data.loc[(data["STREAMINGTV_No internet service"] == 0) & (data["STREAMINGTV_Yes"] == 1), "NEW_STREAMINGTV"] = 1
        data.loc[(data["STREAMINGTV_No internet service"] == 1) & (data["STREAMINGTV_Yes"] == 0), "NEW_STREAMINGTV"] = 0

        data.loc[(data["STREAMINGMOVIES_No internet service"] == 0) & (data["STREAMINGMOVIES_Yes"] == 0), "NEW_STREAMINGMOVIES"] = 0
        data.loc[(data["STREAMINGMOVIES_No internet service"] == 0) & (data["STREAMINGMOVIES_Yes"] == 1), "NEW_STREAMINGMOVIES"] = 1
        data.loc[(data["STREAMINGMOVIES_No internet service"] == 1) & (data["STREAMINGMOVIES_Yes"] == 0), "NEW_STREAMINGMOVIES"] = 0

        data["TOTAL_SERVICE"] = data["NEW_MUTLIPLELINES"] + data["NEW_ONLINESECURITY"] + data["NEW_ONLINEBACKUP"] \
                              + data["NEW_DEVICEPROTECTION"] + data["NEW_TECHSUPPORT"] \
                              + data["NEW_STREAMINGTV"] + data["NEW_STREAMINGMOVIES"] + data["PHONESERVICE_Yes"]

        list = data[["NEW_MUTLIPLELINES", "NEW_ONLINESECURITY", "NEW_ONLINEBACKUP", "NEW_DEVICEPROTECTION",
                   "NEW_TECHSUPPORT", "NEW_STREAMINGTV", "NEW_STREAMINGMOVIES"]]

        data = data.drop(list, axis=1)

        data["TOTALCHARGES_PER_SERVICE"] = data["TOTALCHARGES"] / data["TOTAL_SERVICE"]

        data["TOTALCHARGES_PER_SERVICE"].replace(np.inf, 0, inplace=True)

        data[data["TOTALCHARGES_PER_SERVICE"] == 0].head(10)

        low_mean = data[(data["TOTALCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 0)]["TOTALCHARGES"].mean()
        high_mean = data[(data["TOTALCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 1)]["TOTALCHARGES"].mean()

        data.loc[(data["TOTALCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 0), "TOTALCHARGES_PER_SERVICE"] = low_mean
        data.loc[(data["TOTALCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 1), "TOTALCHARGES_PER_SERVICE"] = high_mean

        data["MONTHLYCHARGES_PER_SERVICE"] = data["MONTHLYCHARGES"] / data["TOTAL_SERVICE"]
        data["MONTHLYCHARGES_PER_SERVICE"].replace(np.inf, 0, inplace=True)

        low_mean = data[(data["MONTHLYCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 0)]["MONTHLYCHARGES"].mean()
        high_mean = data[(data["MONTHLYCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 1)]["MONTHLYCHARGES"].mean()

        data.loc[(data["MONTHLYCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 0), "MONTHLYCHARGES_PER_SERVICE"] = low_mean
        data.loc[(data["MONTHLYCHARGES_PER_SERVICE"] == 0) & (data["PARTNER_Yes"] == 1), "MONTHLYCHARGES_PER_SERVICE"] = high_mean

        zero_mean = data.loc[(data["CONTRACT_One year"] == 0) & (data["CONTRACT_Two year"] == 0)]["TOTALCHARGES"].mean()
        two_mean = data.loc[(data["CONTRACT_One year"] == 0) & (data["CONTRACT_Two year"] == 1)]["TOTALCHARGES"].mean()
        one_mean = data.loc[(data["CONTRACT_One year"] == 1) & (data["CONTRACT_Two year"] == 0)]["TOTALCHARGES"].mean()

        data.loc[(data["CONTRACT_One year"] == 0) & (data["CONTRACT_Two year"] == 0), "CONTRACT_TYPE"] = 1
        data.loc[(data["CONTRACT_One year"] == 0) & (data["CONTRACT_Two year"] == 1), "CONTRACT_TYPE"] = two_mean
        data.loc[(data["CONTRACT_One year"] == 1) & (data["CONTRACT_Two year"] == 0), "CONTRACT_TYPE"] = one_mean

    cat_cols, num_cols, cat_but_car = grab_col_names(data, 10, 20)

    if std_scale:
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

    if rob_scale:
        rs = RobustScaler()
        data[num_cols] = rs.fit_transform(data[num_cols])

    data.drop(columns="CUSTOMERID", inplace=True)
    data.rename(columns = {"CHURN_1": "CHURN"}, inplace=True)

    y = data["CHURN"]
    X = data.drop(["CHURN"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
    rf_model = RandomForestClassifier(random_state=18).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc)

    feature_imp = pd.DataFrame({'Value': rf_model.feature_importances_, 'Feature': X_train.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:len(X)])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


telco_model(df, "CHURN", check_df_df=True, cat_summary_df=True,
                col_summary_df=True, check_outlier_df=True, correlation=True,
                LOF=True, new_features=True, std_scale=True, rob_scale=True)