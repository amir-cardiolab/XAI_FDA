
%Goal is to learn a maping between an input 2D image into another image
%(predict velocity field from permeability field). 

clear all; 
close all;

%% Generate Data

Flag_method =2; % 1 or 2: Two different datasets
Flag_nn_approx = 1; % if 1 use trained NN for probing else use true data. 

if (Flag_method==1)

 N2 =  2250; % 600; %2250; %number of images for training
 N2t = 100;  %number of images for testing
 Flag_downsample = 1; %if 1 downsample training data
 load  porous_train_test_image2image.mat    %../myPorous_NN/myporousCNN_train_test_image2image.mat 
 if (Flag_nn_approx)
  logfile = "Results/logNN_image2image_flow.txt";
  error_file = "Results/errorNN_image2image_flow.mat";
  soln_file = "Results/solnNN_image2image_flow.mat";
  else
  logfile = "Results/logData_image2image_flow.txt";
  error_file = "Results/errorData_image2image_flow.mat";
  soln_file = "Results/solnData_image2image_flow.mat";
 end
else  %Flag_method==2

 N2 =  225; %number of images for training
 N2t = 64;  %number of images for testing
 Flag_downsample = 0; %if 1 downsample training data
 load porousflow_train_test_image2image_flow2.mat 
 if (Flag_nn_approx)
  logfile = "Results/logNN_image2image_flow2.txt";
  error_file = "Results/errorNN_image2image_flow2.mat";
  soln_file = "Results/solnNN_image2image_flow2.mat";
 else
  logfile = "Results/logData_image2image_flow2.txt";
   error_file = "Results/errorData_image2image_flow2.mat";
  soln_file = "Results/solnData_image2image_flow2.mat";
 end

end





N = 28;
N_points = N*N; 
x =linspace(0, 1, N);
y =linspace(0, 1, N);
[x_train1,y_train1] = meshgrid(x,y);
% x_all =linspace(-1., 1., N);
% y_all =linspace(-1., 1., N);
% [x_all,y_all] = meshgrid(x_all,y_all);

% Flag_sparse_data = 1; %If True, adds sparse data in the OOD regime
% if (Flag_sparse_data)
%     x_ood = linspace(-1., 1., 10);
%     x_train1 = [x_train1 x_ood];
% end





if(1)
 Thresh = 1e-3;
 input_images_train(input_images_train<Thresh)  = 0.;
 input_images_ood(input_images_ood<Thresh)  = 0. ;
end

if(0)
input_images_train = input_images_train + 1.;
input_images_ood = input_images_ood + 1.; 
end


if (Flag_downsample==1)
red_fac = 25;    
N2 = N2/ red_fac; 
% N2t = 1;
input_images_train = input_images_train(1:red_fac:end,:,:);
y_predict = y_predict(1:red_fac:end,:);
y_true = y_true(1:red_fac:end,:);
%input_images_ood = input_images_ood(1:N2t,:,:);
%y_predict_ood = y_predict_ood(1:N2t,:);
%y_true_ood = y_true_ood(1:N2t,:);
end

output_train = zeros(N2,N_points);
output_test= zeros(N2t,N_points);

%Read input

index_input_train = linspace(0, 1, N2*N*N);
index_input_test = linspace(0, 1, N2t*N*N);
%f_output = double(y_true');
f_output = double(y_true);
%f_output_test = double(y_true_ood');
f_output_test = double(y_true_ood);
z_input_all =  input_images_train;
z_input_all_test = input_images_ood;

%y_predict = double(y_predict'); %NN prediction
y_predict = double(y_predict); %NN prediction
%y_predict_ood = double(y_predict_ood'); %NN prediction for ood
y_predict_ood = double(y_predict_ood); %NN prediction for ood








%% Reshape 2D arrays
f_output = reshape(f_output',[length(f_output(:)),1]);
f_output_test = reshape(f_output_test',[length(f_output_test(:)),1]);
y_predict = reshape(y_predict',[length(y_predict(:)),1]);
y_predict_ood = reshape(y_predict_ood',[length(y_predict_ood(:)),1]);


if (Flag_nn_approx==1)
y_output = y_predict;
else
y_output = f_output;
end



%% Build library and compute sparse regression


[Theta,eqn_list]  = poolData_image2image_porous_method1(N2,N,z_input_all,x,y,x_train1,y_train1); 


%create a log
if exist(logfile, 'file') ; delete(logfile); end
diary(logfile);

fprintf('condition number of Theta: %e', cond(Theta));





n=1;
if(1) %original method (SINDY)
 lambda = 0.1;
 Xi = sparsifyDynamics(Theta,y_output,lambda,n);
 %Xi = sparsifyDynamics_gmres(Theta,y_output,lambda,n); This  gives a bit worse error
end

if(0) %least squares with L2 regularization  
lambda =1e-9; 
n_terms = size(Theta,2);
%%%Xi =  (transpose(Theta)*Theta + lambda * eye(n_terms) ) \ (transpose(Theta)*y_output)  ;
Xi =  gmres(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output, [], 1e-14,200   ) ; %set tol and maxiter
%Xi =  pcg(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output ) ; 
end



%%% List the equations
fprintf('Equations  \t  Coefficients \r\n');
for i=1:length(Xi)
    fprintf('%s  %.3e \r\n', eqn_list{i}, Xi(i));
end


%%% List the equations


%%% Plot the result
Theta_test = plot_image2image_results_porous(Xi,z_input_all_test,f_output,f_output_test,Theta,N2t,N,x,y,x_train1,y_train1,index_input_test,index_input_train,y_predict,y_predict_ood,N2,error_file, soln_file) ;
 

diary off;


