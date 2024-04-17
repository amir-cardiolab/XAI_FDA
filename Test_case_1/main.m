%Goal is to learn a maping between an input 2D image into a signle value (e.g., learning the strain energy or peak stress given a heterogenous domain stiffness)
%This version: works based on input/output data based on FEniCS simulations
%from MNIST or EMNIST benchmark datasets created by Emma Lejeune's group 


clear all; 
close all;

Flag_nn_approx = 1; % if 1 use trained NN for probing else use true data. 
Flag_downsample = 0; %if 1 downsample training data
Flag_method = 3;  %1 or 2 or 3 for the 3 different datasets.


%% Generate Data
%%Different MNIST and EMNIST datasets
if(Flag_method == 1)
load MNIST_trains100_tests25.mat 
N2 = 2500; %number of images for training
N2t = 2000;  %number of images for testing
if (Flag_nn_approx)
logfile = "Results/logNN_MNIST_trains100_tests25.txt";
error_file = "Results/errorNN_MNIST_trains100_tests25.mat";
soln_file = "Results/solnNN_MNIST_trains100_tests25.mat";
else
logfile = "Results/logData_MNIST_trains100_tests25.txt";
error_file = "Results/errorData_MNIST_trains100_tests25.mat";
soln_file = "Results/solnData_MNIST_trains100_tests25.mat";
end

end

if(Flag_method == 2)
load MNIST_trains100_tests50 
N2 = 2500; %number of images for training
N2t = 2000;  %number of images for testing
if (Flag_nn_approx)
logfile = "Results/logNN_MNIST_trains100_tests50.txt";
error_file = "Results/errorNN_MNIST_trains100_tests50.mat";
soln_file = "Results/solnNN_MNIST_trains100_tests50.mat";
else
logfile = "Results/logData_MNIST_trains100_tests50.txt";
error_file = "Results/errorData_MNIST_trains100_tests50.mat";
soln_file = "Results/solnData_MNIST_trains100_tests50.mat";
end

end

% if(0)
% load MNIST_trainr15_testr1.mat
% N2 = 9800; %number of images for training
% N2t = 2000;  %number of images for testing
% end

if(Flag_method == 3)
load EMNISTshift_trains100_tests10.mat
N2 = 8000; %2500; %number of images for training
N2t = 2000;  %number of images for testing
if (Flag_nn_approx)
logfile = "Results/logNN_EMNISTshift_trains100_tests10.txt";
error_file = "Results/errorNN_EMNISTshift_trains100_tests10.mat";
soln_file = "Results/solnNN_EMNISTshift_trains100_tests10.mat";
else
logfile = "Results/logData_EMNISTshift_trains100_tests10.txt";
error_file = "Results/errorData_EMNISTshift_trains100_tests10.mat";
soln_file = "Results/solnData_EMNISTshift_trains100_tests10.mat";
end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create a log
if exist(logfile, 'file') ; delete(logfile); end
diary(logfile);


N = 28;
x =linspace(0, 1, N);
y =linspace(0, 1, N);
[x_train1,y_train1] = meshgrid(x,y);


output_train = zeros(N2,1);
output_test= zeros(N2t,1);

%Read input
%index_input_train = zeros(N2,1);
index_input_train = linspace(0, 1, N2);
%index_input_test = zeros(N2t,1);
index_input_test = linspace(0, 1, N2t);
%f_output = zeros(N2,1); %final output 
f_output = double(y_true');
%f_output_test = zeros(N2t,1); %final output test
f_output_test = double(y_true_ood');
%z_input_all = zeros(N2,N,N); %all the input images stored
z_input_all =  input_images_train;
%z_input_all_test = zeros(N2t,N,N); %all the input images stored
z_input_all_test = input_images_ood;

y_predict = double(y_predict'); %NN prediction
y_predict_ood = double(y_predict_ood'); %NN prediction for ood




if (Flag_downsample==1)
 N_freq = 10; %Take every N_freq data
 N2 = N2 / N_freq;
 output_train = zeros(N2,1);
 index_input_train = linspace(0, 1, N2);
 f_output = f_output(1:N_freq:end);
 z_input_all = z_input_all(1:N_freq:end,:,:);
 y_predict = y_predict(1:N_freq:end);
end


if (Flag_nn_approx==1)
y_output = y_predict;
else
y_output = f_output;
end

%% Build library and compute sparse regression



[Theta,eqn_list] = poolData_2d_MNIST_method(N2,N,z_input_all,x,y,x_train1,y_train1);  %method4
 

fprintf('condition number of Theta: %e', cond(Theta));


n=1;
if(1) %original method (similar to SINDY)
 lambda = 0.1; 
 Xi = sparsifyDynamics(Theta,y_output,lambda,n);
end

if(0) %least squares with L2 regularization (might work better!)
lambda =1e-9;
n_terms = size(Theta,2);
Xi =  (transpose(Theta)*Theta + lambda * eye(n_terms) ) \ (transpose(Theta)*y_output)  ;
end





%%% List the equations
fprintf('Equations  \t  Coefficients \r\n');
for i=1:length(Xi)
    fprintf('%s  %.3e \r\n', eqn_list{i}, Xi(i));
end




%%% Plot the result
Theta_test = plot_my2D_results_MNIST(Xi,z_input_all_test,f_output,f_output_test,Theta,N2t,N,x,y,x_train1,y_train1,index_input_test,index_input_train,y_predict,y_predict_ood,error_file, soln_file) ;


diary off;




