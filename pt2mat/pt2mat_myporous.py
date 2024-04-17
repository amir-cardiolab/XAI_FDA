import numpy as np
import torch
import vtk
from vtk.util import numpy_support as VN
from torch import nn
import torch.nn.functional as F
#from model import *
#dev = "cuda:0" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
from scipy.io import savemat

#convert the trained pt networks and input/label data to mat format for Matlab

class LeNet_hres(nn.Module):
    def __init__(self):
        super(LeNet_hres, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        #self.conv3= nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1)
        self.conv4= nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4608 , 800)
        self.fc1b = nn.Linear(800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        #x = F.relu(self.conv3(x))
        #x = self.maxpool(x)

        x = F.relu(self.conv4(x))
        x = self.maxpool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1b(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(3136, 1000)
        self.fc1b = nn.Linear(1000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1b(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


dev = "cpu"

fieldname_perm = "kprojected"
fieldname_u = "Vel_mag"

File_input_train = "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/perm_Kall_stenosis_"
File_input_label = "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/Vel_Kall_stenosis_"
num_files = 2250
File_ood = "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/perm_Kall_stenosis_valid_"
File_ood_label =  "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/Vel_Kall_stenosis_valid_"
num_files_ood = 100


Output_file = "myporousCNN_train_test_8020.mat"
nn_file = "../Pytorch_try4/Results/stenosis_CNN2.pt" # 80-20 split revision
File_ood = "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/perm_Kall_stenosis_ood_"  
File_ood_label = "/Users/amir/Data/porous-ML/generalize/FEM/results/vtk_files/Vel_Kall_stenosis_ood_"   


nPt = 64 #28
#nPt = 264 #trying hres
my_eps = 0.001
xStart =  0. + my_eps
xEnd =  1. - my_eps
yStart = 0. + my_eps
yEnd = 1. - my_eps
Norm_factor_perm =  1.74 
Norm_factor_out = 2.

#strcutured grid:
x = np.linspace(xStart, xEnd, nPt)
y = np.linspace(yStart, yEnd, nPt)
x, y = np.meshgrid(x, y)
n_points = nPt * nPt


k_all =  np.zeros((num_files, nPt,nPt))
label_all= np.zeros((num_files,1)) 
k_ood =  np.zeros((num_files_ood, nPt,nPt))
label_ood= np.zeros((num_files_ood,1)) 

for i in range(num_files):

    
    mesh_file = File_input_train + str(i) +".vtu"
    #print ('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk_in = reader.GetOutput()
    
    mesh_file_u = File_input_label  + str(i) +".vtu"
    #print ('Loading', mesh_file_u)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file_u)
    reader.Update()
    data_vtk_out = reader.GetOutput()
    Vel =  VN.vtk_to_numpy(data_vtk_out.GetPointData().GetArray( fieldname_u ))

    if (i ==0):
        VTKpoints = vtk.vtkPoints()
        n = 0
        for j in range(nPt):
            for k in  range(nPt):
              VTKpoints.InsertPoint(n, x[j,k], y[j,k], 0.)
              n = n + 1
        point_data = vtk.vtkUnstructuredGrid()
        point_data.SetPoints(VTKpoints)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk_in)
    #probe.SetSourceData(data_vtk_out)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname_perm)
    #array = probe.GetOutput().GetPointData().GetArray(fieldname_u)
    perm_interped = VN.vtk_to_numpy(array)
    image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_perm
    #image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_out
    k_all[i,:,:] = image_input  
    label_all[i] = np.max(Vel[:]) / Norm_factor_out 


for i in range(num_files_ood):

    
    mesh_file = File_ood + str(i) +".vtu"
    #print ('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk_in = reader.GetOutput()
    
    mesh_file_u = File_ood_label + str(i) +".vtu"
    #print ('Loading', mesh_file_u)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file_u)
    reader.Update()
    data_vtk_out = reader.GetOutput()
    Vel =  VN.vtk_to_numpy(data_vtk_out.GetPointData().GetArray( fieldname_u ))

    if (i ==0):
        VTKpoints = vtk.vtkPoints()
        n = 0
        for j in range(nPt):
            for k in  range(nPt):
              VTKpoints.InsertPoint(n, x[j,k], y[j,k], 0.)
              n = n + 1
        point_data = vtk.vtkUnstructuredGrid()
        point_data.SetPoints(VTKpoints)
    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk_in)
    #probe.SetSourceData(data_vtk_out)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname_perm)
    #array = probe.GetOutput().GetPointData().GetArray(fieldname_u)
    perm_interped = VN.vtk_to_numpy(array)
    image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_perm
    #image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_out
    k_ood[i,:,:] = image_input  
    
    label_ood[i] = np.max(Vel[:]) / Norm_factor_out #max of output field



#######################################

k_all = np.expand_dims(k_all, axis=1)
input_image = torch.Tensor(k_all).to(dev)


print(torch.Tensor.size(input_image))
#plt.imshow(input_image.data.numpy()[2,0,:,:], cmap='rainbow', interpolation='nearest')
#plt.show()


output_label = torch.Tensor(label_all).to(dev)



m=LeNet_hres().to(dev)
m.load_state_dict(torch.load(nn_file,
                             map_location=torch.device(dev)))
m.eval()

y_predict= m(input_image).cpu().detach().numpy() # m(input_image).cpu().detach().numpy().reshape(-1)
y_true = output_label.cpu().detach().numpy()


print (np.shape(y_predict))
print (np.shape(y_true))


k_ood = np.expand_dims(k_ood, axis=1)
test_image = torch.Tensor(k_ood).to(dev)
output_label_ood  = torch.Tensor(label_ood).to(dev)


y_predict_ood = m(test_image).cpu().detach().numpy() #m(test_image).cpu().detach().numpy().reshape(-1)
y_true_ood = output_label_ood.cpu().detach().numpy()



N_train = np.size(y_true)
print ('Number of training data',N_train)
N_test = np.size(y_true_ood)
print ('Number of testing data',N_test)

input_images_train = np.zeros((N_train,nPt,nPt))
input_images_ood = np.zeros((N_test,nPt,nPt))

input_images_train[:,:,:] = input_image.cpu().detach().numpy()[:,0,:,:]
input_images_ood[:,:,:] = test_image.cpu().detach().numpy()[:,0,:,:]

mdic = {"input_images_train": input_images_train, "input_images_ood": input_images_ood, "y_true_ood": y_true_ood, "y_predict": y_predict, "y_true": y_true, "y_predict_ood": y_predict_ood }
savemat(Output_file, mdic)

if(1):
 Error_training = abs( y_predict[:] - y_true[:] )
 Error_ood = abs( y_predict_ood[:] - y_true_ood[:] )
 print ('MAE training data',np.mean(Error_training))
 print ('MAE testing data',np.mean(Error_ood))

 print ('shape',np.shape(Error_ood))
 print ('shape',np.shape(Error_training))


 plt.figure()
 #plt.plot(y_predict,'-', label='NN solution', alpha=1.0,zorder=0)
 #plt.plot( y_true,'r--', label='True solution', alpha=1.0,zorder=0)
 plt.plot(Error_training,'-', label='Error_training', alpha=1.0,zorder=0)
 plt.plot(Error_ood,'r--', label='Error_ood', alpha=1.0,zorder=0)
 #plt.plot(y_predict_ood,'g-', label='NN solution OOD', alpha=1.0,zorder=0)
 #plt.plot( y_true_ood,'g--', label='True solution OOD', alpha=1.0,zorder=0)
 plt.legend(loc='best')
 plt.show()

print('Done!')

