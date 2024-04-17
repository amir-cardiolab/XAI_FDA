function [yout,Equation] = poolData_image2image_porous_method1(N2,N,z_input_all,x,y,x_train1,y_train1)
%image 2 image
%method1: expand method 2 
%selecting the candidate terms

%https://en.wikipedia.org/wiki/Polynomial_kernel
%https://en.wikipedia.org/wiki/Radial_basis_function_kernel
%https://en.wikipedia.org/wiki/Kernel_(statistics)
%https://en.wikipedia.org/wiki/Kernel_density_estimation
%https://en.wikipedia.org/wiki/Radial_basis_function

%https://www.cs.toronto.edu/~duvenaud/cookbook/




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


N_e = 120 ; 
exp_factors =  linspace(0.2,1.5,N_e);



N_t = 18*N_e + 2;  



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
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),') * f dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;
   end


   for j=1:N_e   %http://www.greensfunction.unl.edu/glibcontent/node24.html
         mygreen = exp(- sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(- sqrt((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),') * f dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;
   end




  for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* x_temp; 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int Dist sqrt(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),'/2) * f dxdy'];
           end
           n_green_funcs = n_green_funcs + 1;
  end

   for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* (x_temp.^2); 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int Dist sqrt(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),'/2) * f^2 dxdy'];
           end
           n_green_funcs = n_green_funcs + 1;
   end


  for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* x_temp; 
           yout(i,n_green_funcs) = exp( trapz(y,trapz(x, int_fn'  ,1),2) );
            if(i==1) %save the equation
           Equation{n_green_funcs} = ['exp( \int Dist sqrt(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),'/2) * f dxdy)'];
           end
           n_green_funcs = n_green_funcs + 1;
  end


 for j=1:N_e
           Dist = sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )/ (exp_factors(j)/2);
           mygreen = ones(NN,NN) ; 
           mygreen(Dist>1) = 0;
           int_fn =  mygreen .* exp(x_temp); 
           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
            if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int Dist sqrt(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),'/2) * exp(f) dxdy'];
           end
           n_green_funcs = n_green_funcs + 1;
 end






%          %Polynomial order 1
%          mygreen1 = x_train1(:,:) * xx_train1(ii,jj) + y_train1(:,:) * yy_train1(ii,jj)   ;   %------ 2nd green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen1 .* x_temp; 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%           n_green_funcs = n_green_funcs + 1;
%          
% 
%          %Polynomial order 1
%          mygreen1 = x_train1(:,:) * xx_train1(ii,jj) + y_train1(:,:) * yy_train1(ii,jj)   ;   %------ 2nd green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen1 .* (x_temp.^2); 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%           n_green_funcs = n_green_funcs + 1;



                     %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (x-zeta) /', num2str( exp_factors(j)),') * f dxdy'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (y-eta) /', num2str( exp_factors(j)),') * f dxdy'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

         
                            %%%%%%%%%%%%%%% 

                       %!!!!!!&**************** TEMP. Testing the following two: improved max gen but mean became: 0.0077  &&&&&^^^^^^^^^^^^^^^^^^^^^    
%          for j=1:N_e
%          mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) ).^2  /exp_factors(j));   %------ 22 green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* x_temp; 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%          n_green_funcs = n_green_funcs + 1;
%          end
% 
%                 %%%%%%%%%%%%%%% 
%          for j=1:N_e
%          mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) ).^2  /exp_factors(j));  %------ 23 green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* x_temp; 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%          n_green_funcs = n_green_funcs + 1;
%          end
%         


         % Lets now do nonlinear on input fn with successful Kernels (helps!)    
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^2); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (x-zeta) /', num2str( exp_factors(j)),') * f^2 dxdy'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* (x_temp.^2); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (y-eta) /', num2str( exp_factors(j)),') * f^2 dxdy'];
           end
         n_green_funcs = n_green_funcs + 1;
         end


        %  !!!!!!&**************** Dont help much--> mean gen = 0.0066  ^^^^^^^
%           for j=1:N_e
%          mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* (x_temp.^3); 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%          n_green_funcs = n_green_funcs + 1;
%          end
% 
%                 %%%%%%%%%%%%%%% 
%          for j=1:N_e
%          mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* (x_temp.^3); 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%          n_green_funcs = n_green_funcs + 1;
%          end


        

                               %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* tanh(x_temp); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (x-zeta) /', num2str( exp_factors(j)),') * tanh(f) dxdy'];
          end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =   mygreen .* tanh(x_temp); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['\int exp(- (y-eta) /', num2str( exp_factors(j)),') * tanh(f) dxdy'];
           end
         n_green_funcs = n_green_funcs + 1;
         end
        


         %%%%%%%%% Now apply nonlinearity to output of integral??

         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2)  ^2 ;
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['(\int exp(- (x-zeta) /', num2str( exp_factors(j)),') * f dxdy)^2'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2) ^2;
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['(\int exp(- (y-eta) /', num2str( exp_factors(j)),') * f dxdy)^2'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

        %These twi tanh very important....
         for j=1:N_e
         mygreen =  exp(- (x_train1(:,:) - xx_train1(ii,jj) )  /exp_factors(j));   %------ 22 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2)  ) ;
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['tanh(\int exp(- (x-zeta) /', num2str( exp_factors(j)),') * f dxdy)'];
           end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- (y_train1(:,:) - yy_train1(ii,jj) )  /exp_factors(j));  %------ 23 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2) );
           if(i==1) %save the equation
           Equation{n_green_funcs} = ['tanh(\int exp(- (y-eta) /', num2str( exp_factors(j)),') * f dxdy)'];
           end
         n_green_funcs = n_green_funcs + 1;
         end


         for j=1:N_e   %RBF Kernel
         mygreen = exp(- ( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2) ^ 2;
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['(\int exp(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),') * f dxdy)^2'];
         end
          n_green_funcs = n_green_funcs + 1;
         end

           for j=1:N_e   %RBF Kernel
         mygreen = exp(- ( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(kk,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = tanh (trapz(y,trapz(x, int_fn'  ,1),2) );
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['tanh(\int exp(- ((x-zeta)^2 - (y-eta)^2) /', num2str( exp_factors(j)),') * f dxdy)'];
         end
          n_green_funcs = n_green_funcs + 1;
            end

          % This not much useful  gen mean: 0.0069
%        for j=1:N_e   %http://www.greensfunction.unl.edu/glibcontent/node24.html
%          mygreen = exp(- sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* x_temp; 
%          yout(i,n_green_funcs) =  trapz(y,trapz(x, int_fn'  ,1),2)^2;
%           n_green_funcs = n_green_funcs + 1;
%         end



          %%%%%%%%%%
                %without these gen mean: 0.007
%           mygreen = ( x_train1(:,:) - xx_train1(ii,jj) ).^2 ;
%           int_fn = mygreen .* x_temp ; 
%           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%           n_green_funcs = n_green_funcs + 1;
% 
%           mygreen = ( y_train1(:,:) - yy_train1(ii,jj)).^2 ;
%           int_fn = mygreen .* x_temp ; 
%           yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%           n_green_funcs = n_green_funcs + 1;


%    for j=1:N_e   %Matern Class of Covariance functions (GP ML book)
%          mygreen = (1 + sqrt(6/exp_factors(j)*sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )   ) ) .* exp(- sqrt(6/exp_factors(j)*sqrt( (x_train1(:,:) - xx_train1(ii,jj)).^2 +  (y_train1(:,:) - yy_train1(ii,jj)).^2   )  ));   %------ 2nd green function in the library %%%%%%
%          %x_temp(:,:) = z_input_all(kk,:,:);
%          int_fn =  mygreen .* x_temp; 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
%           n_green_funcs = n_green_funcs + 1;
%    end




                  %%%%%%%%%%%%%%% 
           %------ 24 green function in the library %%%%%% this one just
           %a bias term
%          int_fn =  ones(NN,NN); 
%          yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          yout(i,n_green_funcs) = 1. ; 
          if(i==1) %save the equation
          Equation{n_green_funcs} = ['1'];
          end
          n_green_funcs = n_green_funcs + 1;
% 
% 
          %%%%%
          %the mean of the image
            yout(i,n_green_funcs) = mean(x_temp(:)) ; 
            if(i==1) %save the equation
            Equation{n_green_funcs} = ['\int f dxdy / \int  dxdy'];
            end
            n_green_funcs = n_green_funcs + 1;








       end
   end

end






end