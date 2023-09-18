import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

SFairtraiffic = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')

st.subheader('Time Series Analysis on Air Traffic Passenger Statistics')

# st.write(SFairtraiffic.head())

# Univariant Time Series 
airtraffic = SFairtraiffic[['Activity Period','Passenger Count']]

airtraffic['Activity Period'] = SFairtraiffic['Activity Period'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}")

airtraffic['Activity Period'] = pd.to_datetime(airtraffic['Activity Period'],errors = 'coerce')

# st.write(airtraffic.head())

# Resample Time Series into different Time Frequences
# "60Min" - hour, "D" - Daily, "M" - Monthly, "Q" - Quarterly, "A" - Annual
# Statistical Function typically sum() or mean() or median() must be given. After Resampling Date will be indexed into Row.

airtraffic_monthly = airtraffic.resample("M",on="Activity Period").sum()

tab1,tab2 = st.tabs(["🗃 Data","📈 Line Chart"])

with tab1:
# Top 5 rows of data.
   st.subheader("Top 5 rows of data")
   st.write(airtraffic_monthly.head())


# Plotting Time Series with Plotly
import plotly.express as px
with tab2:
   fig = px.line(airtraffic_monthly, x=airtraffic_monthly.index, y="Passenger Count", title='Air Traffic Passenger Statistics')
   st.plotly_chart(fig)

# Adfuller:
adfuller(airtraffic_monthly)

# KPSS:
kpss(airtraffic_monthly)

airtraffic_monthly_diff = airtraffic_monthly.diff()
airtraffic_monthly_diff = airtraffic_monthly_diff.dropna()

adfuller(airtraffic_monthly_diff)
kpss(airtraffic_monthly_diff)

# Decomposition
decompose = seasonal_decompose(airtraffic_monthly,model="multiplicative")
decompose.plot()
st.pyplot()

# ACF and PACF
plot_acf(airtraffic_monthly)
st.pyplot()

