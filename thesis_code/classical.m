%Sphere Case, N = number of nodes
%N = 10000;
%[p,tri,edge] = getSphere(N);
%Cube Case, 
N = 8^4;
[p,tri,edge] = getCube(N);
%center in origin
[N,~] = size(p);


%Author: Daniel Osen

%Linear basis functions
%running index i, where i=3(j-1)+d, where d is the component and 
%j is the node.

%Case isotropic clamped stationary media, using rubber
%K = 10^9;
%E = 10^4;
%lambda = 3*K*(3*K-E)/(9*K-E);
%mu = (3/2)*(K-lambda);
lambda=2;
mu=2;
%matrix = toeplitz([0.1,0,0,0.05,0,0,0.05,0,0,0.05,0,0]);
% eq. (lambda*m1+(lambda+mu)*m2+mu*m3)u = b
m1 = zeros(12,12);
m2 = zeros(12,12);
mtemp = zeros(12,12);
m_global = zeros(3*N,3*N);
m_global = sparse(m_global);
b_global = zeros(3*N,1);
b_global = sparse(b_global);
v1 = zeros(1,12);
%sphere case
%f_1 = @(x) 2*lambda+8*mu; %solid body forces seperated into components
%f_2 = @(x) 0;
%f_3 = @(x) 0;
%cube case
f_1 = @(x) (-1)*((lambda+2*mu)*2*(x(2)^2-x(2))*(x(3)^2-x(3))+mu*2*(x(1)^2-x(1))*(x(3)^2-x(3))+mu*2*(x(1)^2-x(1))*(x(3)^2-x(3))+(lambda+mu)*(2*x(1)-1)*(2*x(2)-1)*(x(3)^2-x(3))+(lambda+mu)*(2*x(1)-1)*(2*x(3)-1)*(x(2)^2-x(2)));
f_2 = @(x) (-1)*((lambda+2*mu)*2*(x(1)^2-x(1))*(x(3)^2-x(3))+mu*2*(x(2)^2-x(2))*(x(3)^2-x(3))+mu*2*(x(2)^2-x(2))*(x(3)^2-x(3))+(lambda+mu)*(2*x(2)-1)*(2*x(1)-1)*(x(3)^2-x(3))+(lambda+mu)*(2*x(2)-1)*(2*x(3)-1)*(x(1)^2-x(1)));
f_3 = @(x) (-1)*((lambda+2*mu)*2*(x(2)^2-x(2))*(x(1)^2-x(3))+mu*2*(x(3)^2-x(3))*(x(1)^2-x(3))+mu*2*(x(3)^2-x(3))*(x(1)^2-x(1))+(lambda+mu)*(2*x(3)-1)*(2*x(2)-1)*(x(1)^2-x(1))+(lambda+mu)*(2*x(3)-1)*(2*x(1)-1)*(x(2)^2-x(2)));
%L-shape case
[Nk,Np] = size(tri); %Nk number of elements, Np number of points in tetrahedron
Nq = 5; %number of quadrature points
ip = eye(4); %4x4 identity matrix
cp = zeros(4); %4x4 coefficient matrix for linear basis functions
b = zeros(3*4,1); %empty load vector for each tetrahedron using runnning index
u_true = zeros(3*N,1); %empty vector for analytical solution
%sphere case
r_x = @(x) 1-x(1)^2-x(2)^2-x(3)^2; %analytical solution 1st component, the others are 0
%cube case
u_cube = @(x) (x(1)^2-x(1))*(x(2)^2-x(2))*(x(3)^2-x(3));
%%% Construct Stiffness Matrix and Loading Vector %%%
for k=1:Nk
    %grab tetrahedron points corresponding to element with nodes
    %tri(k,:)
    pk = [p(tri(k,:),:),ones(4,1)];
    %solve for linear basis function coefficients
    cp = pk\ip;
    %grab physical volume of tetrahedron
    vk = (1/6)*abs(det([pk(1,1:3)'-pk(4,1:3)',pk(2,1:3)'-pk(4,1:3)',pk(3,1:3)'-pk(4,1:3)']));
    %Construct first matrix
    for i=1:4
        for j=1:4
            %compute integral of shape gradients with quadrature rule,
            %totally unneccesary might as well just scale with volume since
            %these are constant using linear shape functions
            %shape = @(x) cp(1:3,i)'*cp(1:3,j);
            %v1(3*(j-1)+1) = quadrature3D(pk(1,1:3)',pk(2,1:3)',pk(3,1:3)',pk(4,1:3)',Nq,shape);
            v1(3*(j-1)+1) = vk*cp(1:3,i)'*cp(1:3,j);
            
            %construct third matrix in same loop
            m3((3*(i-1)+1):(3*(i-1)+3),(3*(j-1)+1):(3*(j-1)+3)) = vk*[0,cp(1,j)*cp(2,i),cp(1,j)*cp(3,i);
            cp(2,j)*cp(1,i),0,cp(2,j)*cp(3,i);
            cp(3,j)*cp(1,i),cp(3,j)*cp(2,i),0];
        end
        %construct a toeplitz matrix of the integrals, but only use the first three rows
        %done manually
        %mtemp = toeplitz(v1);
        m1((3*(i-1)+1):(3*(i-1)+3),1:12) = [v1;[v1(12),v1(1:11)];[v1(11:12),v1(1:10)]];
    end
    %Construct second matrix
    m2 = vk*[cp(1:3,1);cp(1:3,2);cp(1:3,3);cp(1:3,4)]*[cp(1:3,1)',cp(1:3,2)',cp(1:3,3)',cp(1:3,4)'];
    
    %Construct load
    for i=1:4
        %NB: Anonymous functions in matlab are notoriously slow but...
        %multiply the vector load function with the vector shape function
        %done component wise for each node
        fshape_1 = @(x) f_1(x)*[x(1),x(2),x(3),1]*cp(:,i);
        fshape_2 = @(x) f_2(x)*[x(1),x(2),x(3),1]*cp(:,i); 
        fshape_3 = @(x) f_3(x)*[x(1),x(2),x(3),1]*cp(:,i); 
        %compute the integral for each component using quadrature rule
        %using running index, this is done in physical space
        b(3*(i-1)+1) = quadrature3D(pk(1,1:3)',pk(2,1:3)',pk(3,1:3)',pk(4,1:3)',Nq,fshape_1);
        b(3*(i-1)+2) = quadrature3D(pk(1,1:3)',pk(2,1:3)',pk(3,1:3)',pk(4,1:3)',Nq,fshape_2);
        b(3*(i-1)+3) = quadrature3D(pk(1,1:3)',pk(2,1:3)',pk(3,1:3)',pk(4,1:3)',Nq,fshape_3);
    end
    %add to global matrix, following node order 1,...,N (with 3 entries
    %for each node sine u is a 3d vector field
    m_local = lambda*m1 + (lambda+mu)*m2 + mu*m3;
    pos= 3*(tri(k,:)-1)+1;
    for i=1:4
        for j=1:4
            m_global(pos(i):pos(i)+2,pos(j):pos(j)+2) = m_global(pos(i):pos(i)+2,pos(j):pos(j)+2) + m_local(3*(i-1)+1:3*(i-1)+3,3*(j-1)+1:3*(j-1)+3);
        end
        b_global(pos(i):(pos(i)+2)) = b_global(pos(i):pos(i)+2)+ b((3*(i-1)+1):(3*(i-1)+3));
        %Sphere case
        %u_true(pos(i)) = r_x(pk(i,1:3));
        %cube case
        u_true(pos(i)) = u_cube(pk(i,1:3));
        u_true(pos(i)+1) = u_true(pos(i));
        u_true(pos(i)+2) = u_true(pos(i));
    end
end
%%% PARSE DIRICHlET BOUNDARY
% Get edge boundary and set dirichlet solution
[Ne, Npe] = size(edge);
% here I force the solution to be u=0 on the edges.

for k=1:Ne
    edgepos = 3*(edge(k,:)-1)+1; %grab running index
    for i=1:3
        for j=1:3
            %Idea: set row to 0 and diagonal to 1 for m
            %and to 0 for that index of the load vector, to force that u*1=0
            m_global(edgepos(i)+(j-1),:) = 0;
            %set column to zero, since that u is supposed to be 0
            %maintaining symmetry
            m_global(:,edgepos(i)+(j-1)) = 0;
            %set diagonal to 1
            m_global(edgepos(i)+(j-1),edgepos(i)+(j-1)) = 1;
        end
        b_global(edgepos(i):edgepos(i)+2) = 0;
    end
end
% solve using backslash
u = m_global\b_global;
% write VTK
%string = strcat('classical',num2str(i));
%writeVTK(string,tri,p,full(u(1:3:end)));
%quiver3(p(:,1),p(:,2),p(:,3),u(1:3:end),u(2:3:end),u(3:3:end))
%compare with analytical
u_e= u - u_true;
errors = norm(u_e,inf)
%check if numerical u agrees with analytical u on edges
%x_n = [x_n;N];
%y_n = [y_n;errors];
