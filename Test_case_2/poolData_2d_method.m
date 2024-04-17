function [yout,Equation] = poolData_2d_method(N2,N,z_input_all,x,y,x_train1,y_train1)



N_e = 20; 
exp_factors = linspace(0.1,10,N_e);  





N_t = 8 + 6* N_e ;  %!!!!! update this value if needed
yout = zeros(N2,N_t ); 
x_temp = zeros(N,N);




for i=1:N2
 n_green_funcs = 1;
 x_temp(:,:) = z_input_all(i,:,:);

  %%%%%%%%
  mygreen = ones(N,N);  %------ 1st green function in the library %%%%%%
  int_fn =  mygreen .* x_temp; 
  out0 = trapz(y,trapz(x, int_fn'  ,1),2);
  yout(i,n_green_funcs) = out0;
  if(i==1) %save the equation
  Equation{n_green_funcs} = ['\int f dxdy'];
  end
  n_green_funcs = n_green_funcs + 1;
  %%%%%%%%%%%%
   for j=1:N_e
         mygreen = exp(- ( x_train1(:,:).^2 + y_train1(:,:).^2   ) / exp_factors(j) );   %------ 2nd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(-x^2 - y^2)/', num2str( exp_factors(j)),' * f dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;
   end

  %%%%%%%%%%%%

         mygreen = y_train1(:,:).^2  ;   %------ 3rd green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
          yout(i,n_green_funcs ) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int y^2 * f * dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;
 
     %%%%%%%%%%%%
         mygreen = x_train1(:,:).^2  ;   %------ 4 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
          yout(i,n_green_funcs ) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int x^2 * f * dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;


        %%%%%%%%%%%%

         mygreen = y_train1(:,:)  ;   %------ 5th green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp;  
          yout(i,n_green_funcs ) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int y * f * dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;


     %%%%%%%%%%%%
            mygreen = x_train1(:,:)  ;   %------ 6 green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp;  
          yout(i,n_green_funcs ) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int x * f * dxdy'];
         end
          n_green_funcs = n_green_funcs + 1;
          %%%%%%%%%%%


% 
%         for j=1:N_log
%          mygreen =  log ( sqrt( x_train1(:,:).^2 + y_train1(:,:).^2  ) + log_factors(j) );   %------ 11 green function in the library %%%%%%
%          x_temp(:,:) = z_input_all(i,:,:);
%          int_fn =  mygreen .* x_temp; 
%          yout(i,n_green_funcs )= trapz(y,trapz(x, int_fn'  ,1),2);
%          n_green_funcs = n_green_funcs + 1;
%         end






         mygreen = x_train1(:,:).*y_train1(:,:)  ;   %------ green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs ) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int x*y * f * dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;


                     %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen =  exp(- x_train1(:,:) /exp_factors(j));   %------ green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(-x/',num2str( exp_factors(j)),')* f dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;
         end

                %%%%%%%%%%%%%%% 
         for j=1:N_e
         mygreen = exp(- y_train1(:,:) / exp_factors(j) );   %------ green function in the library %%%%%%
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(-y/',num2str( exp_factors(j)),')* f dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;
         end


                  %%%%%%%%%%%%%%% 
           %------ green function in the library %%%%%% this one just
           %a bias term
         %int_fn =  ones(N,N); 
         yout(i,n_green_funcs) = 1; %trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['1'];
         end
         n_green_funcs = n_green_funcs + 1;

                     %%%%%%%%%%%%%%% 
        %------ green function in the library %%%%%%  !!!!!! THIS WAS
        %THE KEY!!
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  x_temp .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int f^2 * dxdy'];
         end
        n_green_funcs = n_green_funcs + 1;



                         %%%%%%%%%%%%%%% 
        %------ green function in the library %%%%%%  
         for j=1:N_e
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  exp(-x_temp / exp_factors(j)); 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
          if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp(-f/',num2str( exp_factors(j)),') * dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;
         end

                                  %%%%%%%%%%%%%%% 
        %------ green function in the library %%%%%% 
        for j=1:N_e
          mygreen = exp(- ( x_train1(:,:).^2 + y_train1(:,:).^2   ) / exp_factors(j) );  
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp.^2; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2);
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['\int exp( -(x^2 + y^2) /',num2str( exp_factors(j)),') * f^2 * dxdy'];
         end
         n_green_funcs = n_green_funcs + 1;
        end
                                    %%%%%%%%%%%%%%% 
 

                                          %------ green function in the
                                          %library %%%%%%  This does
                                          %improve 
        for j=1:N_e
          mygreen = exp(- ( x_train1(:,:).^2 + y_train1(:,:).^2   ) / exp_factors(j) );  
         %x_temp(:,:) = z_input_all(i,:,:);
         int_fn =  mygreen .* x_temp; 
         yout(i,n_green_funcs) = trapz(y,trapz(x, int_fn'  ,1),2)^2;
         if(i==1) %save the equation
         Equation{n_green_funcs} = ['(\int exp( -(x^2 + y^2) /',num2str( exp_factors(j)),') * f * dxdy)^2'];
         end
         n_green_funcs = n_green_funcs + 1;
        end
          






end



end