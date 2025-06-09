import torch
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
import os, sys
from tqdm.notebook import tqdm_notebook
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.insert(0, '/usr/local/home/rs5cq/research/bb_optimization')

#from src.utils.functions import camel6, bukin6, colville, hart6, shekel_6, griewank, one_x_y2
from src.utils.general import increment_path

# Set the random seed
#seed = 123
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.backends.cudnn.deterministic = True

torch.set_default_dtype(torch.float64)




# Define the class for single layer NN
class one_layer_net(torch.nn.Module):
    # consturctor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons, bias=True)
        #torch.nn.init.zeros_(self.linear_one.weight)
        #torch.nn.init.normal_(self.linear_one.weight,0,10)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size, bias=True)
        #torch.nn.init.zeros_(self.linear_two.weight)
        #self.linear_three = torch.nn.Linear(input_size, output_size, bias=True)
        #self.linear_four = torch.nn.Linear(hidden_neurons, hidden_neurons, bias=True)
        self.relu1 = torch.nn.ReLU()
        
        #self.batch_norm1 = torch.nn.BatchNorm1d(hidden_neurons)
        #torch.nn.init.constant_(self.linear_two.bias, 99400)
        # self.linear_two.weight.data.fill_(1.0)
        # self.linear_two.weight.requires_grad = False

    def forward(self,x):
        return self.linear_two(self.relu1(self.linear_one(x))) #+ self.linear_three(x)
        #return self.linear_two(self.relu1(self.batch_norm1(self.linear_one(x))))
    
    def criterion(self,y_pred, y):
        # out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        #out = torch.mean((y_pred - y)**2)
        #out = torch.mean(torch.abs(y_pred - y))
        out = torch.nn.functional.mse_loss(y_pred,y,reduction='mean')
        return out

    def fit(self,X,Y,epochs,optimizer,scheduler=None):
        cost = []
        pbar = tqdm_notebook(range(epochs),desc = f"Training")
        for epoch in pbar:
            
            # if epoch %10 ==0:
            #     print("inside for debugging")
            optimizer.zero_grad()       
            
            yhat = self(X)
            loss = self.criterion(yhat, Y)
            loss.backward()
            optimizer.step()
                 
            cost.append(loss.item())
            # if cost[-1] == min(cost):
            #     torch.save(self.state_dict(), os.path.join('./nn_models','model.pt'))
                
            if epoch %1000 ==0:
                pbar.set_postfix({'Training Loss': loss.item()})
            if scheduler is not None:
                scheduler.step()
        return cost
    
    def fit_dataloader(self,dataloader,epochs,optimizer,scheduler=None):
        cost = []
        pbar = tqdm_notebook(range(epochs),desc = f"Training")
        for epoch in pbar:
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.to('cuda')
                target = target.to('cuda')
                yhat = self(data)
                loss = self.criterion(yhat, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  #TODO. this should be before loss.backward()            
                cost.append(loss.item())
            if epoch %10 ==0:
                pbar.set_postfix({'Training Loss': loss.item()})
            if scheduler is not None:
                scheduler.step()
        return cost
    
# deep neural network
class DeepNet(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(DeepNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        # Create the hidden layers
        for i in range(len(layer_sizes) - 2):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.activations.append(torch.nn.ReLU())
            #self.activations.append(torch.nn.LeakyReLU(0.1))
            # have batchnormalization layer
            #self.batchnorm = torch.nn.BatchNorm1d(layer_sizes[i+1])

        # Create the output layer
        self.layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x
    
    def criterion(self,y_pred, y):
        # out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        #out = torch.mean((y_pred - y)**2)
        #out = torch.mean(torch.abs(y_pred - y))
        out = torch.nn.functional.mse_loss(y_pred,y,reduction='mean')
        return out
    
    def fit(self,X,Y,epochs,optimizer,scheduler=None):
        cost = []
        pbar = tqdm_notebook(range(epochs),desc = f"Training")
        for epoch in pbar:
            yhat = self(X)
            loss = self.criterion(yhat, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            
            cost.append(loss.item())
            # if cost[-1] == min(cost):
            #     torch.save(self.state_dict(), os.path.join('./nn_models','model.pt'))
                
            if epoch %1000 ==0:
                pbar.set_postfix({'Training Loss': loss.item()})
            if scheduler is not None:
                scheduler.step()
        return cost
    

    
        
# Define the class for single layer NN
class one_layer_net2(torch.nn.Module):
    # consturctor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
    
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size, bias=False)
        self.relu1 = torch.nn.ReLU()

        # self.linear_two.weight.data.fill_(1.0)
        # self.linear_two.weight.requires_grad = False

    def forward(self,x):

        return self.linear_two(self.relu1(self.linear_one(x)))

    def criterion(self,y_pred, y):
        # out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        #out = torch.mean((y_pred - y)**2)
        #out = torch.mean(torch.abs(y_pred - y))
        out = torch.nn.functional.mse_loss(y_pred,y,reduction='sum')
        return out

    def fit(self,X,Y,epochs,optimizer):
        cost = []
        pbar = tqdm_notebook(range(epochs),desc = f"Training")
        for epoch in pbar:
            yhat = self(X)
            loss = self.criterion(yhat, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            
            cost.append(loss.item())
            if epoch %1000 ==0:
                pbar.set_postfix({'Training Loss': loss.item()})
        return None

if __name__ =="__main__":

    function,transform, fmax,x_star = shekel_6()
    d = 4
    # Randomly sampling function

    #hyperparameters
    n_hidden = 100
    learning_rate = 0.001
    epochs = 20000

    project_name =  './nn_training/shekel_6'
    Path(project_name).mkdir(parents=True, exist_ok=True)
    exp_name = 'exp' + f'n_hid_{n_hidden}_n_dim_{d}_lr_{learning_rate}_epoch_{epochs}'
    save_dir = str(increment_path(Path(project_name) / exp_name , exist_ok=False))  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    try:
        data_x = np.load(os.path.join(save_dir,'data_x.npy'))
        data_y = np.load(os.path.join(save_dir,'data_y.npy')).reshape(-1,1)
        print("Loaded data from file")
    except:
        num_samples = 100
        data_x = np.random.uniform(0,1,(num_samples,d))
        data_x = transform(data_x.T).T
        data_y = np.array([function(x) for x in data_x])

        np.save(os.path.join(save_dir,'data_x.npy'),data_x)
        np.save(os.path.join(save_dir,'data_y.npy'),data_y)
        print("Data not found, created new data and saved to file")

    # for hart6 function
    #data_y = function(data_x.T.reshape(500//2,1,6))
    model = one_layer_net(d, n_hidden, 1)  # d is the input dimension, d is the hidden dimension
    X = torch.tensor(data_x).type(torch.FloatTensor)
    Y = torch.tensor(data_y).type(torch.FloatTensor).reshape(len(data_y),1)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.001, lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    print("Number of training samples used: ", len(X))
    model.fit(X,Y,epochs, optimizer)
    directions = model.linear_one.weight.to('cpu').detach().numpy().T
    np.save(os.path.join(save_dir,'directions.npy'),directions)
    # # create the model
    # model = one_layer_net(1,3,1)

    # Sanitiy check
    y_pred = model(X)
    print("Check: ", model.criterion(y_pred,Y))

    # # plt.plot(X.numpy(), model(X).detach().numpy(), 'g',label='Random weights')
    # # optimizer = torch.optim.Adam(model.parameters())
    # # epochs=5000
    # # model.fit(epochs, optimizer)

    if d == 2:

        # # visualize the 2D function and NN output.
        x = np.linspace(0,1,100)
        y = np.linspace(0,1,100)

        xx,yy = np.meshgrid(x,y)
        z = function(transform(np.array([xx,yy])))
        # #z = function(np.array([xx,yy]))
        # # plt.figure(dpi=100)
        plt.figure(figsize=(8, 8), dpi=100)

        plt.imshow(
            z.reshape((len(x), len(y))), 
            origin='lower', 
            #extent=(0, 1,0, 1), 
            extent=(-3, 3,-2, 2),
            cmap='plasma')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'function.png'))


        # # Neural Network output

        # #z_nn = model(torch.tensor(transform(np.array([xx,yy]))).type(torch.FloatTensor).reshape(len(x)*len(y),2)).detach().numpy().reshape(len(x),len(y))
        # #z_nn = model(torch.tensor(np.array([xx,yy])).type(torch.FloatTensor).reshape(len(x)*len(y),2)).detach().numpy().reshape(len(x),len(y))
        z_nn = model(torch.tensor(transform(np.vstack((xx.reshape(100*100),yy.reshape(100*100)))).T).type(torch.FloatTensor)).detach().numpy().reshape(len(x),len(y))
        # z_nn2 = model(X).detach().numpy()
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(
            z_nn,
            origin='lower',
        # extent=(0, 1,0, 1),
            extent=(-3, 3,-2, 2),
            cmap='plasma')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'approximation.png'))




