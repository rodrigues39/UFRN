# # Predict Customer Purchase Behavior
# Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset

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

# EDA
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

df.Age = age_group
df.AnnualIncome = AnnualIncome_group
df.TimeSpentOnWebsite = TimeOnSite_group
df.drop_duplicates(inplace=True)

# Arquitetura da classe
class Architecture(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class

        # We start by storing the arguments as attributes
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Creates the train_step function for our model,
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer

        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename,weights_only=False)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

# Pré-processamento
x = df.drop(columns='PurchaseStatus')
y = df.PurchaseStatus.values

# Dividindo os dados em treino e teste
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)

# padronizando os dados de treino
sc = StandardScaler()

# Aplicando a padronização
sc.fit(x_train)
X_train_sc = sc.transform(x_train)
X_val_sc = sc.transform(x_test)

# convertendo os dados de treinamento e teste em tensores
torch.manual_seed(13)

x_train_tensor = torch.tensor(X_train_sc,dtype=torch.float)
y_train_tensor = torch.tensor(y_train,dtype=torch.float).unsqueeze(1)  ####################### unsqueeze(1), transforma em uma dimensão

x_test_tensor = torch.tensor(X_val_sc,dtype=torch.float)
y_test_tensor = torch.tensor(y_test,dtype=torch.float).unsqueeze(1)


# transformando os tensores em um dataset de tensores
train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
val_dataset = TensorDataset(x_test_tensor,y_test_tensor)

# criando um dataloader
train_loader = DataLoader(dataset=train_dataset,batch_size=20, shuffle=True)
val_loader = DataLoader(dataset=val_dataset,batch_size=20)



# Configuração do modelo
# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.01

torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(x_train_tensor.shape[1], 1))

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a BCE loss function
loss_fn = nn.BCEWithLogitsLoss()



# Treinamento e validação
n_epochs = 300
arch = Architecture(model, loss_fn, optimizer)
arch.set_loaders(train_loader, val_loader)
arch.set_seed(42)
arch.train(n_epochs)


fig = arch.plot_losses()
print(model.state_dict())


# Classification Threshold - Data Visualization
## prediction logits (z), sem passar pela função de saída sigmoíde.
logits_val = arch.predict(X_val_sc[:])

# prediction probabilities
prob_val = torch.sigmoid(torch.as_tensor(logits_val[:]).float())
# transforming the probability to binary
y_pred = (prob_val >= 0.5).to(torch.int8)


cf_matrix = confusion_matrix(y_test,y_pred)
plt.subplots(figsize=(3, 3))
plt.title('Matrix confusion')
sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

plt.show()
print('\n',classification_report(y_test,y_pred))
