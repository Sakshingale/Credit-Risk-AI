import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

sns.heatmap(data.corr(), annot=True)
plt.title("Credit Risk Correlation Matrix")

plt.show()
