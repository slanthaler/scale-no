function [u,B,C] = FEM_array(N_f,k0,betaarray, aarray, varray, farray, garray)
% u solution, B,C energy and L2 matrices, 1/N_f is the mesh size, k0 is the
% wavenumber
tic
x = linspace(0, 1, N_f+1);
y = x;
nNodes = (N_f+1)*(N_f+1);
%sparse assembling
I=zeros(16*N_f^2,1);J=I;K=I;K1=I;K_norm=I;
F=zeros(nNodes, 1);

for i = 1:N_f
    for j = 1:N_f  
        [k,k_norm,f] = elementstiff2_array(x, y, i, j, k0,N_f,betaarray, aarray, varray, farray, garray);
        for p = 1:4
            global_p = loc2glo(N_f, i, j, p);
            for q = 1:4
                index=16*N_f*(i-1)+16*(j-1)+4*(p-1)+q;
                global_q = loc2glo(N_f, i, j, q);
                I(index)=global_p;
                J(index)=global_q;
                K(index)=k(p, q);
                K_norm(index)=k_norm(p, q);
                if p==q
                    K1(index)=1/9/N_f^2;
                elseif p==q+2 || p==q-2
                	K1(index)=1/36/N_f^2;
                else 
                    K1(index)=1/18/N_f^2;
                end
            end
            F(global_p) = F(global_p) + f(p);
        end
    end
end

A=sparse(I,J,K,nNodes,nNodes);
C=sparse(I,J,K1,nNodes,nNodes);
B=sparse(I,J,K_norm,nNodes,nNodes);
u = A.'\F;
result=reshape(u,N_f+1,N_f+1)';
toc
%[X, Y] = meshgrid(x, y);
%figure(1);
%h = surf(X, Y, result);
end