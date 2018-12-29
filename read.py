# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd

data = pd.read_csv("data/us_perm_visas.csv",nrows=100)
for c in data.columns:
	print c
