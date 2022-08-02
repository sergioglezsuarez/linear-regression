from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame({"a": [1,2,3], "b":[4,5, pd.NA]})
print(df.iloc[2]["b"].isna())
