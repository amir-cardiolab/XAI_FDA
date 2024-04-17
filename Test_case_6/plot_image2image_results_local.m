function plot_image2image_results_local(Xi,z_input_all,f_output,Theta,N,x,y,x_train1,y_train1,index_input_train,y_predict,N2_train,error_file,  soln_file)


soln_sindy_train = Theta * Xi;






error_sindy_train = abs(soln_sindy_train - f_output); 
error_NN_train = abs(y_predict- f_output); 
error_wrt_NN = abs(soln_sindy_train - y_predict ); %error of interp model wrt NN


figure;
%plot(index_input_train,f_output);  %training data plot
%hold on;
%plot(index_input_test,f_output_test,'r');  %test data plot
%hold on;
%plot(index_input_train,soln_sindy_train,'k--');
%hold on;
%plot(index_input_test,soln_sindy_test,'g--');

if(0)
[error_NN_test, sortIndex] = sort(error_NN_test);
error_sindy_test = error_sindy_test(sortIndex);
%[error_sindy_test, sortIndex] = sort(error_sindy_test);
%error_NN_test = error_NN_test(sortIndex);
index_input_test = linspace(min(index_input_train),max(index_input_train), length(index_input_test) );

end

if(0)
plot(index_input_train,error_sindy_train,'DisplayName','interp train');  %training data plot
hold on;
plot(index_input_train,error_NN_train,'k--','DisplayName','NN train');
hold off;
legend
end


figure;
databox = [error_NN_train;  error_sindy_train; error_wrt_NN] ;
g = [zeros(length(error_NN_train), 1); ones(length(error_sindy_train), 1); 2*ones(length(error_wrt_NN), 1)  ];
b =boxplot(databox,g,'Labels',{'NN train','Interp train','Interp vs. NN'});
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




fprintf('Mean Abs error interp training %f \n', mean(error_sindy_train) )
fprintf('Mean Abs error NN training %f \n', mean(error_NN_train) )
fprintf('Max Abs error interp training %f \n', max(error_sindy_train) )
fprintf('Max Abs error NN training %f \n', max(error_NN_train) )

fprintf('Mean Abs errorinterp w.r.t NN  %f \n', mean(error_wrt_NN) )
fprintf('Max Abs errorinterp w.r.t NN %f \n', max(error_wrt_NN) )

save(error_file,"error_sindy_train", "error_NN_train","error_wrt_NN" );
save(soln_file,"soln_sindy_train", "f_output", "y_predict" );

if (1)
   timestep = 4; %or any desired

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
