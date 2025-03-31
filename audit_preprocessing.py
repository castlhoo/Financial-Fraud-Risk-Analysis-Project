import pandas as pd
PATH = "accounting_dataset.csv"

doc = pd.read_csv(PATH, header=None)
print(doc)