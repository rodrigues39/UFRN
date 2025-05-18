# importando as bibliotecas
import pandas as pd
import numpy as np

# tratamento e visualização dos dados
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# criação da rede
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict # biblioteca para colocar os pesos referência

# Analise das métricas
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv('/content/customer_purchase_data.csv')
df.head()
df.describe()
df.info()


fig,axs = plt.subplots(3, 3, layout="constrained",figsize=(13, 6))
axs[0,0].hist(df.Age,bins=20)
axs[0,0].set_title('Age')

axs[0,1].hist(df.Gender, bins=20)
axs[0,1].set_title('Gender')

axs[0,2].hist(df.AnnualIncome, bins=20)
axs[0,2].set_title('AnnualIncome')

axs[1,0].hist(df.NumberOfPurchases, bins=20)
axs[1,0].set_title('NumberOfPurchases')

axs[1,1].hist(df.ProductCategory, bins=20)
axs[1,1].set_title('ProductCategory')

axs[1,2].hist(df.TimeSpentOnWebsite, bins=20)
axs[1,2].set_title('TimeSpentOnWebsite')

axs[2,0].hist(df.LoyaltyProgram, bins=20)
axs[2,0].set_title('LoyaltyProgram')

axs[2,1].hist(df.DiscountsAvailed, bins=20)
axs[2,1].set_title('DiscountsAvailed')

axs[2,2].hist(df.PurchaseStatus, bins=20)
axs[2,2].set_title('PurchaseStatus')

plt.show()


plt.figure(figsize=(6, 6))
sns.heatmap(df.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlação entre as variáveis')
plt.show()

# Categorizando os dados
age_group = []
for i in df.Age:
  if i <=20:
    age_group.append(1) # "[18,20]"
  elif i > 20 and i <=30:
    age_group.append(2) # "[21,30]"
  elif i > 30 and i <=40:
    age_group.append(3) # "[31,40]"
  elif i > 40 and i <=50:
    age_group.append(4) #"[41,50]"
  elif i > 50 and i <=60:
    age_group.append(5) #"[51,60]"
  elif i > 60:
    age_group.append(6) #"[60,70]"

AnnualIncome_group = []
for i in df.AnnualIncome:
  if i <30000.0:
    AnnualIncome_group.append(1) #"[20k,30k["
  elif i >=30000.0 and i < 60000.0:
    AnnualIncome_group.append(2) #"[30k,60["
  elif i >=60000.0 and i < 90000.0:
    AnnualIncome_group.append(3) #"[60k,90k["
  elif i >=90000.0 and i < 120000.0:
    AnnualIncome_group.append(4) #"[90k,120k["
  elif i >=120000.0 and i <=150000.0:
    AnnualIncome_group.append(5) #"[120k,150k]"

TimeOnSite_group = []
for i in df.TimeSpentOnWebsite:
  if i <= 10:
    TimeOnSite_group.append(1) #"[1,10]"
  elif i > 10 and i <=20:
    TimeOnSite_group.append(2) #"]10,20]"
  elif i > 20 and i <=30:
    TimeOnSite_group.append(3) #"]20,30]"
  elif i > 30 and i <=40:
    TimeOnSite_group.append(4) #"]30,40]"
  elif i > 40 and i <=50:
    TimeOnSite_group.append(5) #"]40,50]"
  elif i > 50:
    TimeOnSite_group.append(6) #"]50,60]"


# Adicionando a categorização nas features
df.Age = age_group
df.AnnualIncome = AnnualIncome_group
df.TimeSpentOnWebsite = TimeOnSite_group

# Removendo duplicados
df.drop_duplicates(inplace=True )
