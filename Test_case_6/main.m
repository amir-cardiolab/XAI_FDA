
%Goal is to learn a maping between an input 2D image into another image
%(predict velocity field from permeability field). 

clear all; 
close all;

%% Generate Data


Flag_nn_approx = 1; % if 1 use trained NN for probing else use true data. 


 N2 =  100; %number of images for training
 Flag_downsample = 0; %if 1 downsample training data
 load  myporousCNN_train_test_image2image_local.mat  
 if (Flag_nn_approx)
  logfile = "Results/logNN_image2image_local.txt";
  error_file = "Results/errorNN_image2image_local.mat";
  soln_file = "Results/solnNN_image2image_local.mat";
  else
  logfile = "Results/logData_image2image_local.txt";
  error_file = "Results/errorData_image2image_local.mat";
  soln_file = "Results/solnData_image2image_local.mat";
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





if(0)
 Thresh = 1e-3;
 input_images_train(input_images_train<Thresh)  = 0.;
end

if(0)
input_images_train = input_images_train + 1.;
end


if (Flag_downsample==1)
red_fac = 25;    
N2 = N2/ red_fac; 
input_images_train = input_images_train(1:red_fac:end,:,:);
y_predict = y_predict(1:red_fac:end,:);
y_true = y_true(1:red_fac:end,:);

end

output_train = zeros(N2,N_points);


%Read input

index_input_train = linspace(0, 1, N2*N*N);
f_output = double(y_true);
z_input_all =  input_images_train;


y_predict = double(y_predict); %NN prediction








%% Reshape 2D arrays
f_output = reshape(f_output',[length(f_output(:)),1]);
y_predict = reshape(y_predict',[length(y_predict(:)),1]);


if (Flag_nn_approx==1)
y_output = y_predict;
else
y_output = f_output;
end



%% Build library and compute sparse regression


[Theta,eqn_list]  = poolData_image2image_porous_local(N2,N,z_input_all,x,y,x_train1,y_train1); 
%Theta  = poolData_image2image_porous_method1_all(N2,N,z_input_all,x,y,x_train1,y_train1); 

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
plot_image2image_results_local(Xi,z_input_all,f_output,Theta,N,x,y,x_train1,y_train1,index_input_train,y_predict,N2,error_file, soln_file) ;
 

diary off;


