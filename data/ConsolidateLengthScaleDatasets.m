% I have produced a bunch of datasets with Ns512 and Nsamp256.
% Here, I want to consolidate them into several datasets at different length scales.
%
% This script produces:
% 1. training datasets with Ns = 256, 128, 64, 32 and Nsamp = 1024
% 2. test datasets with Ns = 512, 256, 128, 64, 32 and Nsamp = 128

% consolidate training datasets
mode = "";
Nsamp = 256;
Nsamp_desired = 1024;
Ns = [512,256,128,96,64];
ver_start = 1;
consolidate(mode,Nsamp,Nsamp_desired,Ns,ver_start);

% consolidate training datasets
mode = "_test";
Nsamp = 256;
Nsamp_desired = 256;
Ns = [512,256,128,96,64];
ver_start = 1;
consolidate(mode,Nsamp,Nsamp_desired,Ns,ver_start);


function consolidate(mode,Nsamp,Nsamp_desired,Ns,ver_start)
    folder = 'data/';
    fname_pre = sprintf('darcy%s_BC_ver',mode);
    fname_post = sprintf(...
        '_tanh_amin1.0_amax12.0_Nsamp%i_Ns512_alphacoeff1.50_alphag3.50.mat',...
        Nsamp ...
        );
    
    ver = ver_start;
    for s=Ns
        s,ver
        Nsamp = 0;
        while(Nsamp<Nsamp_desired)
            % get "local" data (in current ver)
            filename = strcat(fname_pre,string(ver),fname_post);
            data = load(strcat(folder,filename));
            ToPlot = false;
            data = subsample_data(data,s,ToPlot);
            if Nsamp == 0
                data_glob = data;
            else
                data_glob.input_data = ...
                    cat(1,data_glob.input_data,data.input_data);
                data_glob.output_data = ...
                    cat(1,data_glob.output_data,data.output_data);
                
            end
            ver = ver+1;
            Nsamp = Nsamp+data.Nsamp;
            data_glob.Nsamp = Nsamp;
        end
        % store gathered data
        outfile = sprintf(...
            'darcy%s_BC_tanh_amin1.0_amax12.0_Nsamp%i_Ns%i_alphacoeff1.50_alphag3.50.mat', ...
            mode, data_glob.Nsamp, data_glob.s ...
            );
        save(strcat(folder,outfile),'-struct',"data_glob",'-v7.3');
    end
end

function data = subsample_data(data, s, ToPlot)
    % subsample
    % s = 256; % desired output grid size
    istart = randi([1 data.s-s+1],1); iend = istart+s-1;
    jstart = randi([1 data.s-s+1],1); jend = jstart+s-1;
    
    if ToPlot
        % plot large-scale data
        figure()
        subplot(321)
        pcolor(squeeze(data.input_data(1,1,:,:)))
        colorbar
        shading flat
        title 'coeff'
        hold on
        plot_square(istart,iend,jstart,jend)
        
        subplot(323)
        pcolor(squeeze(data.input_data(1,2,:,:)))
        colorbar
        shading flat
        title 'BC'
        hold on
        plot_square(istart,iend,jstart,jend)
        
        subplot(325)
        pcolor(squeeze(data.output_data(1,:,:)))
        colorbar
        shading flat
        title 'sol'
        hold on
        plot_square(istart,iend,jstart,jend)
    end

    % carry out subsampling
    data.s = s;
    data.input_data = data.input_data(:,:,istart:iend, jstart:jend);
    data.output_data = data.output_data(:,istart:iend, jstart:jend);
    
    % fix boundary data
    data.input_data(:,2,:,:) = data.output_data(:,:,:);
    
    if ToPlot
        % plot subsampled data
        subplot(322)
        pcolor(squeeze(data.input_data(1,1,:,:)))
        colorbar
        shading flat
        title 'coeff'
        
        subplot(324)
        pcolor(squeeze(data.input_data(1,2,:,:)))
        colorbar
        shading flat
        title 'BC'
        
        subplot(326)
        pcolor(squeeze(data.output_data(1,:,:)))
        colorbar
        shading flat
        title 'sol'
    end
end
