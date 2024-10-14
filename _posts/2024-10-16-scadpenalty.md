---
title: Regularization and Variable Selection
categories:
- General
- Assignment 3
excerpt: |
  The smoothly clipped absolute deviation (SCAD) penalty, ElasticNet, and Squareroot Lasso
feature_text: |
  ## Assignment 3
  The smoothly clipped absolute deviation (SCAD) penalty
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---

### SCAD regularization and variable selection
The smoothly clipped absolute deviation (SCAD) penalty was designed to encourage sparse solutions to the least squares problem, while also allowing for large values of β.

The "smoothly clipped" part means that it doesn't penalize large coefficients in a harsh, abrupt way. Instead, it gradually increases the penalty as the coefficient grows larger, but it also "clips" or limits the penalty once the coefficient reaches a certain threshold. This helps balance between preventing overfitting and allowing the model to capture important patterns in the data.

The "absolute deviation" is the magnitude of the difference between the predicted and observed values. It gives us a measure of how much our model's predictions vary from the actual data, without considering the direction of the difference.

$$
f(\beta) = \begin{cases}
\lambda |\beta| & \text{if } |\beta| \leq \lambda \\
a\lambda |\beta| - \frac{\beta^2 - \lambda^2}{a - 1} & \text{if } \lambda < |\beta| \leq a\lambda \\
\frac{\lambda^2}{(a + 1)^2} & \text{if } |\beta| > a\lambda
\end{cases}
$$

Below is my implemented class of SCAD:

```python
class SCAD(nn.Module):
    def __init__(self,input_size, alpha=1.0, bias= True, lambda_val=1.0):

        super(SCAD, self).__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.lambda_val = lambda_val
        self.input_size = input_size
        self.linear = nn.Linear(in_features= input_size,out_features=1,bias=bias,device='cpu',dtype=torch.float64)



    def forward(self, x):
        return self.linear(x)

    def scad_penalty(self, beta):
      abs_beta = torch.abs(beta)
      penalty = torch.zeros_like(beta)

      case1 = abs_beta <=  self.lambda_val
      penalty[case1]= self.lambda_val*abs_beta[case1]
        
        
      case2 = (self.lambda_val < abs_beta) & (abs_beta <= (self.alpha*self.lambda_val))
      penalty[case2]= -(abs_beta[case2]**2 - (2* self.lambda_val*self.alpha*abs_beta[case2])+ self.lambda_val**2)/(2*(self.alpha-1))
      
      case3 = (abs_beta > self.alpha * self.lambda_val)
      penalty[case3]=((self.alpha+1)* self.lambda_val**2)/2

      return penalty.sum()

    def loss(self, y_pred, y_true): 
        """
        Compute the ElasticNet loss function.

        Args:
            y_pred (Tensor): Predicted values with shape (batch_size, 1).
            y_true (Tensor): True target values with shape (batch_size, 1).

        Returns:
            Tensor: The ElasticNet loss.

        """
        mse_loss = nn.MSELoss()(y_pred, y_true)
        scad_penalty = self.scad_penalty(self.linear.weight)
        total_loss = mse_loss + scad_penalty

        return total_loss

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):
        """
        Fit the linear model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        # Define the linear regression layer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.loss(y_pred, y) #can do MSELoss()

            loss.backward() # backward propagation, creating small change for each weight and computes gradient
            optimizer.step() #executing previous step in direction of negative gradient

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict target values for input data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).

        Returns:
            Tensor: Predicted values with shape (num_samples, 1).

        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred
    def get_coefficients(self):
        """
        Get the coefficients (weights) of the linear regression layer.

        Returns:
            Tensor: Coefficients with shape (output_size, input_size).

        """
        return self.linear.weight

```

#### Testing my implementation
I used the [World Happiness Report dataset](https://www.kaggle.com/datasets/unsdsn/world-happiness). I want to select variables based on features' importance.

```python
happiness_data = pd.read_csv('/content/drive/My Drive/DATA440_Capstone/Data Sets/happiness2019.csv')
happiness_data.head()
```


|index|Overall rank|Country or region|Score|GDP per capita|Social support|Healthy life expectancy|Freedom to make life choices|Generosity|Perceptions of corruption|
|---|---|---|---|---|---|---|---|---|---|
|0|1|Finland|7\.769|1\.34|1\.587|0\.986|0\.596|0\.153|0\.393|
|1|2|Denmark|7\.6|1\.383|1\.573|0\.996|0\.592|0\.252|0\.41|
|2|3|Norway|7\.554|1\.488|1\.582|1\.028|0\.603|0\.271|0\.341|
|3|4|Iceland|7\.494|1\.38|1\.624|1\.026|0\.591|0\.354|0\.118|
|4|5|Netherlands|7\.488|1\.396|1\.522|0\.999|0\.557|0\.322|0\.298|

We want to remove the columns 'Country or region' and 'Overall rank'. Score is the target variable. We want to convert the data into tensors so it's compatible with the PyTorch library.

```python
cleandata= happiness_data.drop(['Country or region','Overall rank'],axis=1)
y=torch.tensor(cleandata['Score'].values)
x=torch.tensor(cleandata.drop('Score',axis=1).values)
```

```python
model=SCAD(input_size=x.shape[1])
model.fit(x,y)
```
The output:

```
Epoch [100/1000], Loss: 8.472706216763047
Epoch [200/1000], Loss: 5.832687218891882
Epoch [300/1000], Loss: 5.135207357836537
Epoch [400/1000], Loss: 4.6095757243596704
Epoch [500/1000], Loss: 4.032349200688605
Epoch [600/1000], Loss: 3.7263360875528164
Epoch [700/1000], Loss: 3.4230586672009986
Epoch [800/1000], Loss: 3.1051579907648046
Epoch [900/1000], Loss: 2.8226990444819853
Epoch [1000/1000], Loss: 2.7189077447909478
```

Finally, to look at what variables are important:
```python
importance = model.linear.weight.detach().numpy().flatten()
print("Feature importance based on SCAD regularization: ", importance)
```

The output is:

```
Feature importance based on SCAD regularization:  [ 1.68816155e+00  1.68729348e-04  6.36770909e-04  4.14120192e-03
 -2.12017700e-03 -1.54500250e-04]
```

This demonstrates that all of the variables, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, Perceptions of corruption are significant factors in Happiness scores. Based on the 
# **fix this**

### Experimenting with ElasticNet, SqrtLasso, and SCAD
Let's generate 200 data sets where the input features have a strong correlation structure and apply ElasticNet, SqrtLasso and SCAD to check which method produces the best approximation of an ideal solution. I'll use a "betastar" you design with a sparsity pattern of your choice.

#### ElasticNet
```python
class MyElasticNet(nn.Module):
    def __init__(self,input_size, alpha=1.0, bias= True, l1_ratio=0.5):
        """
        Initialize the ElasticNet regression model.

        Args:
            input_size (int): Number of input features.
            alpha (float): Regularization strength. Higher values of alpha
                emphasize L1 regularization, while lower values emphasize L2 regularization.
            l1_ratio (float): The ratio of L1 regularization to the total
                regularization (L1 + L2). It should be between 0 and 1.

        """
        super(MyElasticNet, self).__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.input_size = input_size
        self.linear = nn.Linear(in_features= input_size,out_features=1,bias=bias,device='cpu',dtype=torch.float64)



    def forward(self, x):
        """
        Forward pass of the ElasticNet model.

        Args:
            x (Tensor): Input data with shape (batch_size, input_size).

        Returns:
            Tensor: Predicted values with shape (batch_size, 1).

        """
        return self.linear(x)

    def objfunc(self, y_pred, y_true): #formerly loss
        """
        Compute the ElasticNet loss function.

        Args:
            y_pred (Tensor): Predicted values with shape (batch_size, 1).
            y_true (Tensor): True target values with shape (batch_size, 1).

        Returns:
            Tensor: The ElasticNet loss.

        """
        mse_loss = nn.MSELoss()(y_pred, y_true)
        l1_reg = torch.norm(self.linear.weight, p=1)
        l2_reg = torch.norm(self.linear.weight, p=2)

        objective = (1/2) * mse_loss + self.alpha * (
            self.l1_ratio * l1_reg + (1 - self.l1_ratio) * (1/2)*l2_reg**2)

        return objective

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):
        """
        Fit the ElasticNet model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        # Define the linear regression layer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.objfunc(y_pred, y) #can do MSELoss()

            loss.backward() # backward propagation, creating small change for each weight and computes gradient
            optimizer.step() #executing previous step in direction of negative gradient

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict target values for input data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).

        Returns:
            Tensor: Predicted values with shape (num_samples, 1).

        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred
    def get_coefficients(self):
        """
        Get the coefficients (weights) of the linear regression layer.

        Returns:
            Tensor: Coefficients with shape (output_size, input_size).

        """
        return self.linear.weight
```
#### Squareroot Lasso
```python
class MySqrtLasso(nn.Module):
    def __init__(self, input_size, alpha=0.1):
        """
        Initialize the  regression model.


        """
        super(MySqrtLasso, self).__init__()
        self.input_size = input_size
        self.alpha = alpha


        # Define the linear regression layer
        self.linear = nn.Linear(input_size, 1,bias=False,device=device,dtype=torch.float64)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input data with shape (batch_size, input_size).

        Returns:
            Tensor: Predicted values with shape (batch_size, 1).

        """
        return self.linear(x)

    def loss(self, y_pred, y_true):
        """
        Compute the loss function.

        Args:
            y_pred (Tensor): Predicted values with shape (batch_size, 1).
            y_true (Tensor): True target values with shape (batch_size, 1).

        Returns:
            Tensor: The loss.

        """
        mse_loss = nn.MSELoss()(y_pred, y_true)
        l1_reg = torch.norm(self.linear.weight, p=1,dtype=torch.float64)

        loss = torch.sqrt(mse_loss) + self.alpha * (l1_reg)

        return loss

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):
        """
        Fit the model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict target values for input data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).

        Returns:
            Tensor: Predicted values with shape (num_samples, 1).

        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred

    def get_coefficients(self):
        """
        Get the coefficients (weights) of the linear regression layer.

        Returns:
            Tensor: Coefficients with shape (output_size, input_size).

        """
        return self.linear.weight
```
Function to make correlated features:
```python
def make_correlated_features(num_samples,p,rho,betastar,noise_std=0.12):
  vcor = []
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  #generating target with noise
  noise = np.random.normal(0, noise_std, size=num_samples)
  y = X @ betastar + noise
  return X,y
```
```python
beta =np.array([-1,1,3,0,2,4])
beta = beta.reshape(-1,1)
betastar = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)
num_samples = 100
rho = 0.9 
p=20

x, y = make_correlated_features(num_samples,p,rho,betastar)
```
Viewing the correlation between our features:

```python
#Creating subplots
mu = np.mean(x,axis=0)

fig, ax = plt.subplots(nrows=3, ncols=2,figsize=(6,8))


plt.subplot(3,2,1)
plt.plot(x[:,0], x[:,1], 'b.')
plt.plot(mu[0], mu[1], 'ro')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.axis('equal')
plt.grid(True)

plt.subplot(3,2,2)
plt.plot(x[:,0], x[:,2], 'b.')
plt.plot(mu[0], mu[2], 'ro')
plt.xlabel('x[0]')
plt.ylabel('x[2]')
plt.axis('equal')
plt.grid(True)

plt.subplot(3,2,3)
plt.plot(x[:,0], x[:,3], 'b.')
plt.plot(mu[0], mu[3], 'ro')
plt.xlabel('x[0]')
plt.ylabel('x[3]')
plt.axis('equal')
plt.grid(True)

plt.subplot(3,2,4)
plt.plot(x[:,1], x[:,2], 'b.')
plt.plot(mu[1], mu[2], 'ro')
plt.xlabel('x[1]')
plt.ylabel('x[2]')
plt.axis('equal')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(x[:,1], x[:,3], 'b.')
plt.plot(mu[1], mu[3], 'ro')
plt.xlabel('x[1]')
plt.ylabel('x[3]')
plt.axis('equal')
plt.grid(True)

plt.subplot(3,2,6)
plt.plot(x[:,2], x[:,3], 'b.')
plt.plot(mu[2], mu[3
                   ], 'ro')
plt.xlabel('x[2]')
plt.ylabel('x[3]')
plt.axis('equal')
plt.grid(True)
fig.tight_layout()
plt.show()
cols = ['X1','X2','X3','X4','X5']

#Viewing the correlation matrix between the first 5 features
sns.heatmap(np.corrcoef(np.transpose(x[:,:5])),cmap='bwr',vmin=-1,vmax=1,annot=True,fmt='.2f',annot_kws={"size": 7},xticklabels=cols,yticklabels=cols)
plt.show()
```
#**insert images**

Comparing 
```python
elastic_net_errors = []
sqrt_lasso_errors = []
scad_errors = []

n_datasets = 200

for _ in range(n_datasets):

  X, y = make_correlated_features(num_samples,p,rho, betastar)
  if torch.cuda.is_available():
    device='cuda' 
  else:
    device= 'cpu'

  X = torch.tensor(x,device=device)
  y = torch.tensor(y,device=device)

  #ElasticNet
  elastic_net_model = ElasticNet(input_size=X.shape[1], alpha=0.1, l1_ratio=0.5)
  elastic_net_model.fit(X, y, num_epochs=1000, learning_rate=0.01)
  
  beta_en = elastic_net_model.get_coefficients().detach().numpy().flatten()
  en_error = mse(betastar, beta_en)
  elastic_net_errors.append(en_error)

  #SqrtLasso
  sqrtlasso_model=SqrtLasso(input_size=X.shape[1],alpha=0.1)
  sqrtlasso_model.fit(X,y,num_epochs=1000,learning_rate=0.01)

  beta_sl = sqrtlasso_model.get_coefficients().detach().numpy().flatten()
  sl_error = mse(betastar, beta_sl)
  sqrt_lasso_errors.append(sl_error)
                      
  #SCAD
  scad_model = SCAD(input_size=X.shape[1],alpha=0.1)
  scad_model.fit(X, y)

  beta_scad = scad_model.get_coefficients().detach().numpy().flatten()
  scad_error = mse(betastar, beta_scad)
  scad_errors.append(scad_error)

print("ElasticNet mean error:", np.mean(elastic_net_errors))
print("SqrtLasso mean error:", np.mean(sqrt_lasso_errors))
print("SCAD mean error:", np.mean(scad_errors))
```

The output:
```
ElasticNet mean error: 3.0669821328959745
SqrtLasso mean error: 1.585806390195204
SCAD mean error: 3.779851160895123
```

### Applying to the Concrete dataset with quadratic interaction terms
Using the methods above, I want to determine a variable selection for the Concrete data set with quadratic interaction terms (polynomial features of degree 2). To solve this, you should consider choosing the best weight for the penalty function. What is the ideal model size (number of variables with non-zero weights), and what is the cross-validated mean square error?

### References


