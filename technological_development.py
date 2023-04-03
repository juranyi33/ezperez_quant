import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np


# importing USA average corn yields
df_usa = pd.read_csv(
    r'C:\Users\JuranyiD\projektek\ezperez_quant\84\agri\average-corn-yields-in-the-united-states.csv').rename(
    {'Corn yield (USDA, 2023)': 'yield_us'}, axis=1)
df_usa['yield_us'] = df_usa['yield_us'].rolling(window=5).mean()

#importing Minnesota data, (competition)
df_min = pd.read_csv(
    r'C:\Users\JuranyiD\projektek\ezperez_quant\84\agri\minnesota_county_yearly_agricultural_production.csv').rename(
    {'YIELD, MEASURED IN BU / ACRE': 'yield_min'}, axis=1)
df_min = df_min[df_min['Commodity'] == 'CORN'][['Year', 'County', 'Crop', 'yield_min']].dropna()
df_min = df_min[['Year', 'yield_min']].groupby(['Year'])['yield_min'].mean()
df_min = df_min.rolling(window=5).mean()

fig = make_subplots(specs=[[{"secondary_y": True}]])

#Plot Minny data next to USA data if neccesary to see the correlation between the two
# fig.add_trace(
#     go.Scatter(x=df_min.index, y=df_min, name="Minnesota corn yield (bu/acre), MA (window=5)"),
#     secondary_y=False,
# )

# add the trace of USA MA yield data
fig.add_trace(
    go.Scatter(x=df_usa['Year'], y=df_usa['yield_us'], name="USA corn yield (tonnes/hectare), MA (window=5)"),
    secondary_y=True,
)

# fig.show()

#separate 3 periods, constant in the begging, linear1 between 1940-55, linear2 between 1956-2022
year_masks = {0: (df_usa['Year'] < 1940), 1: (df_usa['Year'] >= 1940) & (df_usa['Year'] <= 1955),
              2: (df_usa['Year'] > 1955)}
# split the data into three parts: constant, linear1, linear2
linear_data1 = df_usa[year_masks[1]].reset_index(drop=True)
linear_data2 = df_usa[year_masks[2]].reset_index(drop=True)

#measure the trend, and add a starting point to it, which is the ending point of the prev period
def trend_array(data, intercept):
    # Fit linear regression model
    X = data['Year'].values.reshape(-1, 1)
    y = data['yield_us'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)

    # Express best-fit curve as a function of the first year
    m = reg.coef_[0][0]
    f = lambda x: m * (x - data['Year'][0]+1) + intercept

    array = data['Year'].apply(f)
    return array


constant_0 = pd.Series([df_usa[year_masks[0] == True]['yield_us'].mean()] * len(df_usa[year_masks[0] == True]))
trend_1 = trend_array(linear_data1, constant_0.iloc[-1])
trend_2 = trend_array(linear_data2, trend_1.iloc[-1])

#merge the trends into one, which than could be used to devide the target variables with
tech_dev = np.concatenate([constant_0, trend_1, trend_2])
data = pd.Series(tech_dev, index=df_usa['Year'])

#compare the trends with the moving average of USA corn yield
fig.add_trace(
    go.Scatter(x=df_usa['Year'], y=tech_dev, name="trend"),
    secondary_y=False,
)

#devide the yield by the trend to get a "trend-free" output
# fig.add_trace(
#     go.Scatter(x=df_usa['Year'], y=df_usa['yield_us']/tech_dev, name="trend"),
#     secondary_y=False,
# )

fig.show()
