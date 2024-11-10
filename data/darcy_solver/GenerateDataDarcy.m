%% generate data for the Darcy flow problem
% to be used in symmetry NO project

%
Nsamp = 1024;
s = 128;

alpha_coeffs = [6,4,3,2,1];
alpha_gs = [6,4,3,2,1];

for i=1:5
  %
  alpha_coeff = alpha_coeffs(i); % was =2
  alpha_g = alpha_gs(i);     % was =2
  tau = 2.;
  
  fprintf('alpha coeff/g: %g / %g\n',alpha_coeff,alpha_g);

  for k=1:2
    if k==1
       mode = '';
    else
      mode = '_test';
    end
    fprintf('mode = ',mode);

    %%
    %
    input_data = zeros(Nsamp,2,s,s);
    output_data = zeros(Nsamp,s,s);


    %
    fprintf('Generating input data ...\n')
    for i=1:Nsamp
      % generate coefficient field
      input_data(i,1,:,:) = exp( GRF(alpha_coeff, tau, s) );
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
    filename = sprintf('darcy%s_BC_Nsamp%i_Ns%i_alphacoeff%.2f_alphag%.2f.mat',mode,Nsamp,s,alpha_coeff,alpha_g);
    fprintf('Storing the data in %s', filename);
    save([folder,filename], ...
	 "input_data","output_data","alpha_coeff", ...
	 "alpha_g","s","Nsamp","tau", ...
	 '-v7.3');
  end
end
