function Theta_test =plot_image2image_results_porous(Xi,z_input_all_test,f_output,f_output_test,Theta,N2,N,x,y,x_train1,y_train1,index_input_test,index_input_train,y_predict,y_predict_ood,N2_train,error_file,  soln_file)


soln_sindy_train = Theta * Xi;


[Theta_test, eqn_list] = poolData_image2image_porous_method1(N2,N,z_input_all_test,x,y,x_train1,y_train1);   




soln_sindy_test = Theta_test * Xi;

error_sindy_train = abs(soln_sindy_train - f_output); 
error_sindy_test = abs(soln_sindy_test- f_output_test); 
error_NN_train = abs(y_predict- f_output); 
error_NN_test = abs(y_predict_ood- f_output_test); 

figure;
%plot(index_input_train,f_output);  %training data plot
%hold on;
%plot(index_input_test,f_output_test,'r');  %test data plot
%hold on;
%plot(index_input_train,soln_sindy_train,'k--');
%hold on;
%plot(index_input_test,soln_sindy_test,'g--');

if(1)
[error_NN_test, sortIndex] = sort(error_NN_test);
error_sindy_test = error_sindy_test(sortIndex);
%[error_sindy_test, sortIndex] = sort(error_sindy_test);
%error_NN_test = error_NN_test(sortIndex);
index_input_test = linspace(min(index_input_train),max(index_input_train), length(index_input_test) );

end


plot(index_input_train,error_sindy_train,'DisplayName','interp train');  %training data plot
hold on;
plot(index_input_test,error_sindy_test,'r','DisplayName','interp OOD');  %test data plot
hold on;
plot(index_input_train,error_NN_train,'k--','DisplayName','NN train');
hold on;
plot(index_input_test,error_NN_test,'g--','DisplayName','NN OOD');
ylabel('MAE error');
hold off;
legend

figure;
databox = [error_NN_train;  error_sindy_train; error_NN_test; error_sindy_test] ;
g = [zeros(length(error_NN_train), 1); ones(length(error_sindy_train), 1); 2*ones(length(error_NN_test), 1); 3*ones(length(error_sindy_test), 1)   ];
b =boxplot(databox,g,'Labels',{'NN train','Interp train','NN test','Interp test'});
ylabel('MAE')
title('Mean absolute error (MAE) distribution')
ax = gca; 
ax.FontSize = 18; 
linesM = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(linesM, 'Color', [0.4660 0.6740 0.1880]);
lines = findobj(gcf, 'type', 'line');
set(lines, 'LineWidth', 3);
% markers = findobj(gcf,'Tag',  'Outliers');
% set(markers, 'Color', 'm');
%set(markers, 'MarkerSize', 4);

fprintf('Mean Abs error interp generalization %f \n', mean(error_sindy_test) )
fprintf('Mean Abs error NN generalization %f \n', mean(error_NN_test) )
fprintf('Max Abs error interp generalization %f \n', max(error_sindy_test) )
fprintf('Max Abs error NN generalization %f \n', max(error_NN_test) )


fprintf('Mean Abs error interp training %f \n', mean(error_sindy_train) )
fprintf('Mean Abs error NN training %f \n', mean(error_NN_train) )
fprintf('Max Abs error interp training %f \n', max(error_sindy_train) )
fprintf('Max Abs error NN training %f \n', max(error_NN_train) )

save(error_file,"error_sindy_train", "error_sindy_test", "error_NN_train", "error_NN_test" );
save(soln_file,"soln_sindy_train", "f_output", "y_predict" );

if (1)
   timestep = 2; %or any desired

   figure;
   soln_reshaped = reshape(f_output,[N,N,N2_train]);
   soln = soln_reshaped(:,:,timestep);
   MIN = min(soln(:));
   MAX = max(soln(:));
   soln1 = reshape(soln_reshaped(:,:,timestep),[N,N]);
   %contourf(x_train1,y_train1,soln1','LineColor','none');
   pcolor(soln1'); hold on;
   shading interp;
   colormap(jet(256));
   colorbar;
   caxis([MIN,  MAX]);
   title('True solution');
   ax = gca; 
   ax.FontSize = 18; 
   set(gca,'XTick',[], 'YTick', [])

   figure;
   soln_reshaped = reshape(soln_sindy_train,[N,N,N2_train]);
   soln1 = reshape(soln_reshaped(:,:,timestep),[N,N]);
   %contourf(x_train1,y_train1,soln1');
   pcolor(soln1'); hold on;
   shading interp;
   colormap(jet(256));
   colorbar;
   caxis([MIN, MAX]);
   title('Interpretable model solution');
   ax = gca; 
   ax.FontSize = 18; 
   set(gca,'XTick',[], 'YTick', [])
   

   
   figure;
   soln_reshaped = reshape(y_predict,[N,N,N2_train]);
   soln1 = reshape(soln_reshaped(:,:,timestep),[N,N]);
   %contourf(x_train1,y_train1,soln1');
   pcolor(soln1'); hold on;
   shading interp;
   colormap(jet(256));
   colorbar;
   caxis([MIN,  MAX]);
   title('Neural network solution');
   ax = gca; 
   ax.FontSize = 18; 
   set(gca,'XTick',[], 'YTick', [])
 
    
    

end
