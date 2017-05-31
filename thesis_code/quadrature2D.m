function I = quadrature2D(p1,p2,p3,Nq,g)
%QUADRATURE2D Computes 2D-Numerical integration using Gaussian Quadrature.
%             Takes real arguments: (p1,p2,p3,Nq,g), where
%             p1,p2,p3 are triangle cornerpoints of element, 
%             Nq {1,3,4} is number of quadrature points,
%             and g is the function handle to be integrated.
%             Returns the value of the integral I.

%Convert to column vectors if needed
chckrow = size(p1);
if chckrow(1)==1
    p1=p1';
    p2=p2';
    p3=p3';
end
%Area of physical triangle
A_k = (1/2)*abs(det([p2-p1 p3-p1]));
%Mapping from ref. element using linear nodal shapes
Z = @(xi,eta) p1 + [p2-p1, p3-p1]*[xi;eta];
%Assume g handles vector input
switch Nq
    case 1
        z1 = Z(1/3,1/3);
        I = A_k*g(z1);
    case 3
        z1 = Z(0,1/2);
        z2 = Z(1/2,0);
        z3 = Z(1/2,1/2);
        I = A_k*(1/3)*(g(z1)+g(z2)+g(z3));
    case 4
        z1 = Z(1/3,1/3);
        z2 = Z(1/5,3/5);
        z3 = Z(1/5,1/5);
        z4 = Z(3/5,1/5);
        I = -9/16*g(z1) +25/48*(g(z2)+g(z3)+g(z4));
        I = I*A_k;
    otherwise
        I = NaN;
end
end

