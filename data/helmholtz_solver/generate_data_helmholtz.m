%% generate data for the Helmholtz equation
% to be used in symmetry NO project



%
k_list = [5,10,25,50,100];
Nsamp = 1024;
s_list = [64,128,256,512,1024];

%
alpha_a = 1.5;
alpha_g = 3.5; % smoothness of BC ~ smoothness of solution(?)
tau = 2.;

%
transform_choice = 'lognormal';
if strcmp(transform_choice,'lognormal')
    eps = 5;
    amin = 0;
    amax = 0;
    transform = @(coeff_field) exp(coeff_field/eps); % log_normal
elseif strcmp(transform_choice,'tanh')
    eps = 1/100;
    amin = 1;
    amax = 12; 
    transform = @(coeff_field) amin + 0.5*(amax-amin)*(1 + tanh(coeff_field/eps));
else
    error(sprintf('transform_choice %s not recognized.',transform_choice))
end

fprintf('alpha coeff/g: %g / %g\n',alpha_a,alpha_g);

tic;
for dataset=4:4 % generate a bunch of similar datasets (afterwards will subsample to generate datasets with longer length scales)
    k = k_list(dataset);
    s = s_list(dataset);
    for m=1:1
      %
      if m==1
        mode = '';
      else
        mode = '_test';
      end
  
      fprintf('mode = %s\n',mode);
  
      %%
      %
      input_data = zeros(Nsamp,2,s+1,s+1);
      output_data = zeros(Nsamp,s+1,s+1);
  
      %
      fprintf('Generating input data ...\n')

      for i=1:Nsamp
        % generate coefficient field
        a = transform( GRF(alpha_a, tau, 2*s+1) );
        g = GRF(alpha_g, tau, 2*s+1);
        
        input_data(i,1,:,:) = a(1:2:end,1:2:end);
        input_data(i,2,:,:) = g(1:2:end,1:2:end);
        
        output_data(i,:,:) = helmholtz(s, k, a, g);
        
        if mod(i,round(Nsamp/100)) == 0
            toc;
            fprintf('dataset %i, wavenumber %i, instance %i, progress %3i %%\n', dataset, k, i, round(100*i/Nsamp))
            tic;
        end
        
      end
  
      %% storing the data
      folder = 'data/';
      filename = sprintf('helmholtz%s_BC_k%i_%s_amin%.1f_amax%.1f_Nsamp%i_Ns%i_alphacoeff%.2f_alphag%.2f.mat', ...
                         mode, k, transform_choice, amin, amax, Nsamp, s+1, alpha_a, alpha_g);
      fprintf('Storing the data in %s', filename);
      save([folder,filename], ...
  	 "input_data","output_data","alpha_a", ...
  	 "alpha_g","s","Nsamp","tau", ...
	 "dataset", "amax", "amin", "eps", "transform_choice", ...
  	 '-v7.3');
    end
end
toc;

%%%
%figure()
%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.8, 0.2, 0.9, 0.4]);
%
%subplot(131)
%pcolor(squeeze(input_data(1,1,:,:)))
%shading flat
%title input
%colorbar
%
%subplot(132)
%pcolor(squeeze(input_data(1,2,:,:)))
%shading flat
%title BC
%colorbar
%
%
%subplot(133)
%pcolor(squeeze(output_data(1,:,:)))
%shading flat
%title output
%colorbar
