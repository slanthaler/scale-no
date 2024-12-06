function u = helmholtz(N,k, aarray, garray)
% betaarray (2N+1,2N+1), aarray(N,N), varray(N,N), farray(N,N), garray(2N+1,2N+1)

if ~exist('betaarray', 'var') || isempty(betaarray)
  
    betaarray = zeros(2*N+1, 2*N+1);
    betaarray(1,:)=1;betaarray(end,:)=1;betaarray(:,1)=1;betaarray(:,end)=1;
    % aarray = ones(2*N+1, 2*N+1);
    varray = ones(2*N+1, 2*N+1);
    farray = zeros(2*N+1, 2*N+1);
    % garray = zeros(2*N+1, 2*N+1);
end

garray(2:end-1,2:end-1) = 0;

% u = FEM(N, k);
u = FEM_array(N, k, betaarray, aarray, varray, farray, garray);
u = reshape(u, [N+1, N+1]);

end