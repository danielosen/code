function [I] = robinsurfquad3D(p_e,ce,Nq,g)
%AREAQUAD3D computes surface integral over a triangle using 2D gauss quad.
%Takes arguments (pe,ce,Nq,g) where (pe) are points giving the triangle,
%(ce) are the coeffs. of the nodal shape functions,
% and (g) is the the function handle. Nq is # quad points {1,3,4}
%Returns integral for nodes i=1:length(ce)
p1 = p_e(1,:)';
p2 = p_e(2,:)';
p3 = p_e(3,:)';
[rows cols] = size(ce);
%Area of triangle
n = cross(p2-p1,p3-p1);
A_k = (1/2)*dot(n,n)^(1/2);

%1. Map from 2D unit triangle to proj. physical triangle in x-y plane,
%   [x;y] = [x1;y1] + [x2-x1,x3-x1;y2-y1,y3-y1][xi;eta]
%2. Insert this mapping into the surface equation z=z(x,y)
%3. Stare in wonder at how this gives the same result as mapping:
%   [p1 p2 p3][lambda1;lambda2;lambda3] = [x;y;z] (barycentric coords.)
%4. Also, the z-density*area of projected triangle gives the area of the
%   physical triangle (which is obvious actually).
I = zeros(cols,1);
Z = @(w1,w2,w3) w1*p1+w2*p2+w3*p3;
for i=1:cols
switch Nq
    case 1
        z1 = Z(1/3,1/3,1/3);
        I(i) = A_k*g(z1)*[1 z1']*ce(:,i);
    case 3
        z1 = Z(1/2,1/2,0);
        z2 = Z(1/2,0,1/2);
        z3 = Z(0,1/2,1/2);
        I(i) = A_k*(1/3)*(g(z1)*[1 z1']*ce(:,i)+g(z2)*[1 z2']*ce(:,i)+g(z3)*[1 z3']*ce(:,i));
    case 4
        z1 = Z(1/3,1/3,1/3);
        z2 = Z(3/5,1/5,3/5);
        z3 = Z(1/5,3/5,1/5);
        z4 = Z(1/5,1/5,3/5);
        I(i) = -9/16*g(z1)*[1 z1']*ce(:,i) +25/48*(g(z2)*[1 z2']*ce(:,i)+g(z3)*[1 z3']*ce(:,i)+g(z4)*[1 z4']*ce(:,i));
        I(i) = I(i)*A_k;
    otherwise
        I(i) = NaN;
end
end
end

