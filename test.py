import pandas as pd
import re
import unicodedata
# Doc va in ra cac dong dau
df = pd.read_csv('sentiment140.csv', encoding="latin-1", header=None)
print(df.sample(5))