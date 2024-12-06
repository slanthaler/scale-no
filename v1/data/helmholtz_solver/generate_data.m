


N = 1000; % grid size
k = 100; % wavenumber

alpha_a = 1.5;
alpha_g = alpha_a + 2; % smoothness of BC ~ smoothness of solution(?)
tau = 2.;

x = linspace(0,1,2*N+1);
y = linspace(0,1,2*N+1);
[X, Y] = meshgrid(x, y);



% betaarray (2N+1,2N+1), aarray(N,N), varray(N,N), farray(N,N), garray(2N+1,2N+1)
betaarray = zeros(2*N+1, 2*N+1);
betaarray(1,:)=1;betaarray(end,:)=1;betaarray(:,1)=1;betaarray(:,end)=1;
% aarray = ones(2*N+1, 2*N+1);
aarray = exp(GRF(alpha_a, tau, 2*N+1));
varray = ones(2*N+1, 2*N+1);
farray = zeros(2*N+1, 2*N+1);
garray = zeros(2*N+1, 2*N+1);

garray = GRF(alpha_g, tau, 2*N+1);
garray(2:end-1,2:end-1) = 0;

% for i=0:1/(2*N):1
%    for j=0:1/(2*N):1
%       if (i-0.5)^2+(j-0.5)^2<1/400
%           i_index = round(i*2*N)+1;
%           j_index = round(j*2*N)+1;
%           farray(i_index, j_index) = 10000*exp(-1/(1-400*((i-0.5)^2+(j-0.5)^2)));
%       end
%    end
% end

% u = FEM(N, k);
u = FEM_array(N, k, betaarray, aarray, varray, farray, garray);
u = abs(u);
u = reshape(u, [N+1, N+1]);

imagesc(u);
colorbar;