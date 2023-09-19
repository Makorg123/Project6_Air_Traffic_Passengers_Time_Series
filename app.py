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

st.subheader('Time Series Analysis on :red[San Francisco Air Traffic Passenger] Statistics')

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

# Difference plotting
fig = px.line(airtraffic_monthly_diff, x=airtraffic_monthly_diff.index, y="Passenger Count", title='Stationary plot')
st.plotly_chart(fig)

tab3,tab4 = st.tabs(["ACF","PACF"])
# ACF and PACF
with tab3:
   fig2 = plot_acf(airtraffic_monthly_diff,lags=20)
   st.pyplot(fig2)

with tab4:
   fig3 = plot_pacf(airtraffic_monthly_diff,lags=20)
   st.pyplot(fig3)

# Arima model
arima_model = auto_arima(airtraffic_monthly, start_p=0, start_q = 0)

# arima_model.summary()
# SARIMAX(0,1,0) = Non Seasonal Model
# AIC - 6279.860

arima_model.predict(n_periods=24)

index_of_fc = pd.date_range(airtraffic_monthly.index[-1],periods =24,freq='M')

arimapredict =  pd.DataFrame(arima_model.predict(n_periods=24))

arimapredict.index = index_of_fc

# Arima model plotting
fig = px.line(arimapredict, x=arimapredict.index, y=0, title='Arima Model')
st.plotly_chart(fig)