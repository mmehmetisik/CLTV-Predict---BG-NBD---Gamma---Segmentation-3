import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


order = pd.read_csv(r"D:\DataScience\Python\Miuul_Data_Science\Hard_Skill\Hard_Skill_Ödevler\free_weeks\cltv\order.csv")
order.head(5)

customer = pd.read_csv(r"D:\DataScience\Python\Miuul_Data_Science\Hard_Skill\Hard_Skill_Ödevler\free_weeks\cltv\customer.csv")
customer.head(5)

ff=pd.merge(order,customer,on="order_id", how = "inner" )
ff.head(5)

ff["order_purchase_timestamp"] = pd.to_datetime(ff["order_purchase_timestamp"])

ff["order_purchase_timestamp"].max()

today_date = dt.datetime(2018, 9, 5)

ff.groupby('customer_id')

cltv_df = ff.groupby('customer_id').agg(
                    {'order_purchase_timestamp': [lambda order_purchase_timestamp: (order_purchase_timestamp.max() - order_purchase_timestamp.min()).days,
                                                  lambda order_purchase_timestamp: (today_date - order_purchase_timestamp.min()).days],
                    'order_id': lambda x: x.nunique(),
                    'payment_value': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head(2)

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

cltv_df.head(5)


# Establishment of BG-NBD Model

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# Who are the customers we expect to purchase the most during the week?

cltv_df["expected_purc_4_weeks"] = bgf.predict(4,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

cltv_df.head(5)
cltv_df.sort_values(by='expected_purc_4_weeks', ascending=False).head(5)

#Establishing the GAMMA-GAMMA Model
# Determination of coefficients

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# Expected average profit

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(5)


# Calculation of CLTV with BG-NBD and GG model


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 months
                                   freq="W",  # Frequency information of T
                                   discount_rate=0.01)
cltv = cltv.reset_index()
cltv.head(5)

cltv_final = cltv_df.merge(cltv, on="customer_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Creating Segments by CLTV

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10).reset_index()

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})












