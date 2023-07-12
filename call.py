import pandas as pd
from test import logistic_regression
df=pd.read_csv('ready_df')


logistic_regression(df)