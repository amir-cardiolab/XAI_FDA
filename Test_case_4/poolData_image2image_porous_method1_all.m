function yout = poolData_image2image_porous_method1_all(N2,N,z_input_all,x,y,x_train1,y_train1)
%image 2 image
%method1: expand method 2 
%selecting the candidate terms

%https://en.wikipedia.org/wiki/Polynomial_kernel
%https://en.wikipedia.org/wiki/Radial_basis_function_kernel
%https://en.wikipedia.org/wiki/Kernel_(statistics)
%https://en.wikipedia.org/wiki/Kernel_density_estimation
%https://en.wikipedia.org/wiki/Radial_basis_function

%https://www.cs.toronto.edu/~duvenaud/cookbook/


Flag_normalize = 0;  %if 1, normalize the Kernel
Flag_meancenter = 0; %if 1, then mean-center the input data

if(0)
 NN = 10; %number of pts used in enforcing the images being equal < N 
 xx =linspace(0, 1, NN);
 yy =linspace(0, 1, NN);
 [xx_train1,yy_train1] = meshgrid(xx,yy);
else
  xx_train1 = x_train1;
  yy_train1 = y_train1;
  NN = N;
end

if (Flag_meancenter)
mean_all = mean(z_input_all);
z_input_all = z_input_all - mean_all;
end



N_e =  120;
exp_factors = linspace(0.1,1.9,N_e); % linspace(0.2,1.5,N_e);

N_t = 23*N_e + 5;  



yout = zeros(N2*NN*NN,N_t ); % (number of files * total number of pt,  number of integral terms)
x_temp = zeros(NN,NN);

%%%%%%

%int\ G(x,xi)*f(xi)dxi




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



i=0;
for kk=1:N2
   kk 
   x_temp(:,:) = z_input_all(kk,:,:);
   for ii=1:NN
       for jj =1:NN
  i = i + 1;
  n_green_funcs = 1;

  %%%%%%%%%%%%
   for j=1:N_e   %RBF Kernel
         mygreen = exp(- ( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
         end

          n_green_funcs = n_green_funcs + 1;
   end

   for j=1:N_e   %http://www.greensfunction.unl.edu/glibcontent/node24.html
         mygreen = exp(- sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
         end
          n_green_funcs = n_green_funcs + 1;
   end




  for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* x_temp; 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
            end
           n_green_funcs = n_green_funcs + 1;
  end

   for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* (x_temp.^2); 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
            end
           n_green_funcs = n_green_funcs + 1;
   end


  for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* x_temp; 
           yout(i,n_green_funcs) = exp( trapz(y,trapz(x, int_fn'  ,1),2) );
             if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
            end
           n_green_funcs = n_green_funcs + 1;
  end


 for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* exp(x_temp); 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
             if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
            end
           n_green_funcs = n_green_funcs + 1;
 end






         %Polynomial order 1
         mygreen1 = x_train1(:,:) * xx_train1(ii,jj) + y_train1(:,:) * yy_train1(ii,jj)   ;   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen1 .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen1'  ,1),2);
            end
          n_green_funcs = n_green_funcs + 1;
         

         %Polynomial order 1
         mygreen1 = x_train1(:,:) * xx_train1(ii,jj) + y_train1(:,:) * yy_train1(ii,jj)   ;   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen1 .* (x_temp.^2); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if (Flag_normalize)
             yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen1'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;



                     %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end

         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

         
                            %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) ).^2  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) ).^2  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end
        


         % Lets now do nonlinear on input fn with successful Kernels (helps!)    
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^2); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^2); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end



          for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^3); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^3); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end


        

                               %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* tanh(x_temp); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =   mygreen .* tanh(x_temp); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end
        


         %%%%%%%%% Now apply nonlinearity to output of integral??

         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2)  ^2 ;
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2) ^2;
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end


          for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2)  ) ;
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2) );
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
         n_green_funcs = n_green_funcs + 1;
         end


         for j=1:N_e   %RBF Kernel
         mygreen = exp(- ( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2) ^ 2;
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;
         end

           for j=1:N_e   %RBF Kernel
         mygreen = exp(- ( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2) );
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;
            end


       for j=1:N_e   %http://www.greensfunction.unl.edu/glibcontent/node24.html
         mygreen = exp(- sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) =  trapz(y,trapz(x, int_fn'  ,1),2)^2;
            if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;
   end



          %%%%%%%%%%
          mygreen = ( x_train1(:,:) - xx_train1(ii,jj) ).^2 ;
          int_fn = mygreen .* x_temp ; 
          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
             if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;

          mygreen = ( y_train1(:,:) - yy_train1(ii,jj)).^2 ;
          int_fn = mygreen .* x_temp ; 
          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
             if (Flag_normalize)
            yout(i,n_green_funcs) = yout(i,n_green_funcs) / trapz(y,trapz(x, mygreen'  ,1),2);
          end
          n_green_funcs = n_green_funcs + 1;

                  %%%%%%%%%%%%%%% 
           %------ 24 green function in the library %%%%%% this one just
           %a bias term
%          int_fn =  ones(NN,NN); 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          yout(i,n_green_funcs) = 1. ; 
          n_green_funcs = n_green_funcs + 1;
% 
% 
          %%%%%
          %the mean of the image
            yout(i,n_green_funcs) = mean(x_temp(:)) ; 
            n_green_funcs = n_green_funcs + 1;








       end
   end

end






end