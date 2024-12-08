%% Test the solve with method of manufactured solutions
%

Nvals = [64,128,256,512,1024];
errors = [];

% parameters for reference solution
k = randi([1,6]);
m = randi([1,6]);
alpha = 2*rand();
beta = 2*rand();

%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(s-1)
for s=Nvals

    %Create mesh (only needed for plotting)
    [X,Y] = meshgrid(0:(1/(s-1)):1);
    
    % define solution u(x,y)
    u = cos(k*X).*sin(m*Y);
    u1 = -k*sin(k*X).*sin(m*Y);
    u2 = +m*cos(k*X).*cos(m*Y);
    
    % define coefficient field coeff(x,y)
    exy = exp(-alpha*X - beta*Y);
    coeff = 1 + exy;
    coeff1 = -alpha*exy;
    coeff2 = -beta*exy;
    
    % define RHS
    % F = -coeff * d^2u - dcoeff * du.
    F = (k^2 + m^2)*coeff.*u - coeff1.*u1 - coeff2.*u2;
    
    % solve for u via solver
    u_sol = solve_gwf(coeff,F,u);

    % compute error 
    errors(end+1) = norm(u-u_sol)/s;
end


%% plot convergence study
figure()
loglog(Nvals,errors,'*-', ...
       Nvals,Nvals.^(-2),'k--')
legend('computed', 'N^{-2}')
grid()
xlabel('N')
ylabel('error')

%% compare the two
figure()
subplot(1,3,1)
pcolor(X,Y,u)
colorbar
shading flat
title('analytical solution')

subplot(1,3,2)
pcolor(X,Y,u_sol)
colorbar
shading flat
title('numerical solution')

subplot(1,3,3)
pcolor(X,Y,abs(u-u_sol))
colorbar
shading flat
title('difference')
