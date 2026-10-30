import pandas as pd
import numpy as np
df = pd.read_csv("books.csv")

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary:\n", df.describe().round(2))

print("\nCategory-wise avg price & rating:")
print(df.groupby("category").agg(books=("title","count"),
      avg_price=("price","mean"), avg_rating=("rating","mean")).round(2))

print("\nIn-stock %:", round(df.in_stock.mean()*100, 1))
print("Price vs Rating correlation:", round(df.price.corr(df.rating), 3))

# Outlier detection (IQR method)
q1, q3 = df.price.quantile([.25, .75]); iqr = q3 - q1
out = df[(df.price < q1-1.5*iqr) | (df.price > q3+1.5*iqr)]
print("Price outliers:", len(out))

# Hypothesis test: is the average price different for in-stock vs out-of-stock books?
from scipy import stats
a = df[df.in_stock].price; b = df[~df.in_stock].price
t, p = stats.ttest_ind(a, b, equal_var=False)
print(f"\nT-test p-value = {p:.3f} ->", "Difference is significant" if p < 0.05 else "No significant difference")