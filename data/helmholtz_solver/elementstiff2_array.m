function [K,K_norm,f] = elementstiff2_array(X, Y, m, n, k0, N, betaarray, aarray, varray, farray, garray)
%assembling the local stiffness matrix, pay attention to boundary where we 
%need to incorporate robin bc with beta and g

% betaarray (2N+1,2N+1), aarray(N,N), varray(N,N), farray(N,N), garray(2N+1,2N+1)

K = zeros(4, 4);
K_norm = zeros(4, 4);
vs = cellVertices(X, Y, m, n);
xlow = vs(1, 1);
xhigh = vs(3, 1);
ylow = vs(1, 2);
yhigh = vs(3, 2);
f=zeros(4,1);
x = (xlow+xhigh)/2;
y = (ylow+yhigh)/2;

x_index = round(2*N*x)+1; y_index = round(2*N*y)+1; 
xlow_index = round(2*N*xlow)+1; ylow_index = round(2*N*ylow)+1;
xhigh_index = round(2*N*xhigh)+1; yhigh_index = round(2*N*yhigh)+1;

% disp(x_index);disp(xlow_index);disp(xhigh_index);

length = -xlow+xhigh;

beta = [betaarray(x_index,ylow_index)+betaarray(xlow_index,y_index), ...
    betaarray(x_index,ylow_index)+betaarray(xhigh_index,y_index), ...
    betaarray(x_index,yhigh_index)+betaarray(xhigh_index,y_index),...
    betaarray(x_index,yhigh_index)+betaarray(xlow_index,y_index)];
beta1 = [betaarray(x_index,ylow_index),betaarray(xhigh_index,y_index),betaarray(x_index,yhigh_index)];
for i = 1:4 
    for j = 1:4
        if i==j
            K_norm(i,j)=2/3*aarray(x_index,y_index)+k0^2*varray(x_index,y_index)^2*1/9*length^2;
            K(i,j)=2/3*aarray(x_index,y_index)-k0^2*varray(x_index,y_index)^2*1/9*length^2-1i*k0*beta(i)*1/3*length;
        elseif i==j+2 || i==j-2
            K_norm(i,j)=-1/3*aarray(x_index,y_index)+k0^2*varray(x_index,y_index)^2*1/36*length^2;
            K(i,j)=-1/3*aarray(x_index,y_index)-k0^2*varray(x_index,y_index)^2*1/36*length^2;
        elseif i==j+1 || i==j-1
            index = min(i,j);
            K_norm(i,j)=-1/6*aarray(x_index,y_index)+k0^2*varray(x_index,y_index)^2*1/18*length^2;
            K(i,j)=-1/6*aarray(x_index,y_index)-k0^2*varray(x_index,y_index)^2*1/18*length^2-1i*k0*beta1(index)*1/6*length;
        else
            K_norm(i,j)=-1/6*aarray(x_index,y_index)+k0^2*varray(x_index,y_index)^2*1/18*length^2;
            K(i,j)=-1/6*aarray(x_index,y_index)-k0^2*varray(x_index,y_index)^2*1/18*length^2-1i*k0*betaarray(xlow_index,y_index)*1/6*length;
        end
    end
end

f(1) = farray(x_index,y_index)*length^2/4+(garray(x_index,ylow_index)+garray(xlow_index,y_index))*length/2;
f(2) = farray(x_index,y_index)*length^2/4+(garray(x_index,ylow_index)+garray(xhigh_index,y_index))*length/2;
f(3) = farray(x_index,y_index)*length^2/4+(garray(x_index,yhigh_index)+garray(xhigh_index,y_index))*length/2;
f(4) = farray(x_index,y_index)*length^2/4+(garray(x_index,yhigh_index)+garray(xlow_index,y_index))*length/2;

end