
%Goal is to learn a maping between an input 2D image into a signle value (learning the max velocity mag in heterogenous porous media flow)



clear all; 
close all;

%% Generate Data

load myporousCNN_train_test_8020.mat



Flag_nn_approx =1; % if 1 use trained NN for probing else use true data. 
Flag_downsample = 0; %if 1 downsample training data

if (Flag_nn_approx)
logfile = "Results/logNN_scalar_porous.txt";
error_file = "Results/errorNN_scalar_porous.mat";
soln_file = "Results/solnNN_scalar_porous.mat";
else
logfile = "Results/logData_scalar_porous.txt";
error_file = "Results/errorData_scalar_porous.mat";
soln_file = "Results/solnData_scalar_porous.mat";
end

%create a log
if exist(logfile, 'file') ; delete(logfile); end
diary(logfile);

N =  64;
x =linspace(0, 1, N);
y =linspace(0, 1, N);
[x_train1,y_train1] = meshgrid(x,y);


N2 = 2250; % 600; %2250; %number of images for training
N2t = 100;  %number of images for testing

output_train = zeros(N2,1);
output_test= zeros(N2t,1);

%Read input

index_input_train = linspace(0, 1, N2);
index_input_test = linspace(0, 1, N2t);
f_output = double(y_true);
f_output_test = double(y_true_ood);
z_input_all =  input_images_train;
z_input_all_test = input_images_ood;

y_predict = double(y_predict); %NN prediction
y_predict_ood = double(y_predict_ood); %NN prediction for ood




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



[Theta,eqn_list] = poolData_2d_method(N2,N,z_input_all,x,y,x_train1,y_train1); 


fprintf('condition number of Theta: %e', cond(Theta));


n=1;
if(0) %original method (SINDY)
 lambda = 0.1; 
 Xi = sparsifyDynamics(Theta,y_output,lambda,n);
end

if(1) %least squares with L2 regularization (works better!)
lambda =1e-9; 
n_terms = size(Theta,2);
%Xi =  (transpose(Theta)*Theta + lambda * eye(n_terms) ) \ (transpose(Theta)*y_output)  ;
%Xi =  gmres(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output , [], 1e-10,200   ) ; %set tol and maxiter
%Xi =  gmres(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output ) ; 
Xi =  pcg(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output ) ;  %Works very well
%Xi = bicgstab(transpose(Theta)*Theta + lambda * eye(n_terms), transpose(Theta)*y_output, 1e-10,200 ) ; %Note: doing very accurate like this reduces Green training error (makes it similar to NN) but increases generalization error.
%Xi = lsqr(Theta,y_output);
end


%%% List the equations
fprintf('Equations  \t  Coefficients \r\n');
for i=1:length(Xi)
    fprintf('%s  %.3e \r\n', eqn_list{i}, Xi(i));
end




%%% Plot the result
Theta_test = plot_my2D_results(Xi,z_input_all_test,f_output,f_output_test,Theta,N2t,N,x,y,x_train1,y_train1,index_input_test,index_input_train,y_predict,y_predict_ood,error_file, soln_file) ;


diary off;

