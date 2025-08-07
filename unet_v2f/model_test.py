from fileinput import filename
import torch 
import sys
import random 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from torchvision.io import decode_image
import numpy as np 
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
outfile = TemporaryFile()
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import TensorDataset, DataLoader
from Stair_nn_v1 import Stair_nn
from unet_v2a import Dataset
from unet_v2a import Stair_UNet




## creating a dataloader as an object that loads data from .npy file
## codes related to class Data_set() directly copied from Stari_nn_v1.py as a data_loader
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Data_Set():
    def __init__(self,inpfilename,ansfilename):
        inp_numpy = np.load(inpfilename)
        ans_numpy = np.load(ansfilename)
        self.tensor_inp = torch.from_numpy(inp_numpy).to(device).float() 
        self.tensor_ans = torch.from_numpy(ans_numpy).to(device).float()

    def __len__(self):
        return len(self.tensor_inp)
    

    def __getitem__(self,idx):
        return self.tensor_inp[idx], self.tensor_ans[idx] ### it returns the ith set of data by calling __getitem__()




class Model_Test():
    def __init__(self, model_name, inpfilename, ansfilename):
        self.inp_numpy = np.load(inpfilename)
        self.model_name = model_name
        self.inpfilename = inpfilename
        self.ansfilename = ansfilename 
        data = Dataset(inpfilename, ansfilename)
        self.data_loader = DataLoader(data,batch_size = 5)
 

        ##load model from .tp file 
        model = torch.load(self.model_name, weights_only=False)
        y = model(data[97][0].unsqueeze(0)) 
        y_np = y[0].cpu().detach().numpy()  ## switching tensor back to np array

        ## print output y as np array 
        self.printdata(y_np)
        print("__________________________________________________________________")
        self.printdata(self.inp_numpy[0])
        self.showdata(y_np)

        ## run showdata(groundtrueth)
        ## run showdata(modeloutput)

    ## construct a showdata() function that prints and output a graph of the cloud-point map given the np.array
    ## This def function needs a whole lot of changes!!!



  
    def printdata(self,y_np):
        self.full_array_string = np.array2string(np.rint(y_np.reshape((64,64))), threshold = sys.maxsize)
        self.full_array_string = np.array2string(y_np, threshold = sys.maxsize)
        ## self.x_length = y_np.shape[0]
        ## self.y_length = y_np.shape[1] // curently not working because y_np is structured in a  1*2500 np array
        print(self.full_array_string)

    def showdata(self, y_np):
        fig = plt.figure(figsize=(50, 50))
        ax = fig.add_subplot(111, projection='3d')
        plt.show()

        cnt = 0
        for i in range(self.x_length):
            for j in range(self.x_length):
                ax.scatter(i, j, round(y_np[cnt]/1), color='blue', marker='o', s=50, alpha=0.8)
                cnt +=1 
        plt.show()




## argument format: "folder_name/file_name"
MT = Model_Test("unet_v2/unet_nn_v2a_t.pt", "test_data2/generated_inp_test2.npy", "test_data2/generated_ans_test2.npy")



print("it went through")

