import pandas as pd
import numpy as np


xl = pd.ExcelFile('~/Downloads/ex3d1.xlsx')
df = pd.read_excel(xl, 'X', header=None)
y = pd.read_excel(xl, 'y', header = None)
df_numpy = df.to_numpy()
y_numpy = y.to_numpy()
df.to_csv('~/Downloads/X.csv', sep=',', header=False, index=False)
y.to_csv('~/Downloads/y.csv', sep=',', header=False, index=False)


