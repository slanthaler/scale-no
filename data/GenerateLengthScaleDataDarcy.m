%% generate data for the Darcy flow problem
% to be used in symmetry NO project

% Given a length scale L (integer), I produce data in the following manner
% 1. Generate data at length scale 1 (this is the most fine-grained data)
% 2. Randomly subsample on a subdomain of length ~ 1/L, and take this as 
%    input data
% 3. Compute corresponding solution 
% 
% That's it. In this way, all datasets at various length scales are
% "compatible", in the sense that subsampling from one correspondings to 
% sampling from the other (in distribution).

%
Nsamp = 256;
s = 512;

%
alpha_coeff = 1.5;
alpha_g = alpha_coeff + 2; % smoothness of BC ~ smoothness of solution(?)
tau = 2.;

%
transform_choice = 'tanh';
if strcmp(transform_choice,'lognormal')
    eps = 5;
    amin = 0;
    amx = 0;
    transform = @(coeff_field) exp(coeff_field/eps); % log_normal
elseif strcmp(transform_choice,'tanh')
    eps = 1/100;
    amin = 1;
    amax = 12; 
    transform = @(coeff_field) amin + 0.5*(amax-amin)*(1 + tanh(coeff_field/eps));
else
    error(sprintf('transform_choice %s not recognized.',transform_choice))
end

%
length_scales = [1,2,4,8,16];
fprintf('alpha coeff/g: %g / %g\n',alpha_coeff,alpha_g);

for version=20:-1:15 % generate a bunch of similar datasets (afterwards will subsample to generate datasets with longer length scales)
    for k=2:-1:1
      %
      if k==1
        mode = '';
      else
        mode = '_test';
      end
  
      fprintf('mode = %s\n',mode);
  
      %%
      %
      input_data = zeros(Nsamp,2,s,s);
      output_data = zeros(Nsamp,s,s);
  
      %
      fprintf('Generating input data ...\n')
      for i=1:Nsamp
        % generate coefficient field
        input_data(i,1,:,:) = transform( GRF(alpha_coeff, tau, s) );
        input_data(i,2,:,:) = GRF(alpha_g, tau, s);
      end
  
      fprintf('Solving for the solutions ... (This may take some time)\n')
      f = zeros(s,s);
      %
      for i=1:Nsamp
        % solve for u
        coeff = squeeze(input_data(i,1,:,:));
        g     = squeeze(input_data(i,2,:,:));
  
        %
        output_data(i,:,:) = solve_gwf(coeff,f,g);
        if mod(i,round(Nsamp/100)) == 0
  	fprintf('%3i %%\n', round(100*i/Nsamp))
        end
      end
  
      %% storing the data
      folder = 'data/';
      filename = sprintf('darcy%s_BC_ver%i_%s_amin%.1f_amax%.1f_Nsamp%i_Ns%i_alphacoeff%.2f_alphag%.2f.mat', ...
                         mode, version, transform_choice, amin, amax, Nsamp, s, alpha_coeff, alpha_g);
      fprintf('Storing the data in %s', filename);
      save([folder,filename], ...
  	 "input_data","output_data","alpha_coeff", ...
  	 "alpha_g","s","Nsamp","tau", ...
	 "version", "amax", "amin", "eps", "transform_choice", ...
  	 '-v7.3');
    end
end

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
