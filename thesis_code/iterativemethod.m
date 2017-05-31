%Iterative non-local method
%Author: Daniel Osen
N = 200;
[p,tri,edge] = getSphere(N);

%Grab number of elements Nk, NP=4
[Nk,Np] = size(tri);
[N_edge,Np_edge] = size(edge);
%Pre-Allocate matrices
ip = eye(4);
id = eye(3);
%M_global = sparse(zeros(3*N,3*N));
B_global = zeros(3*N,1);
u_true = zeros(3*N,1);
list_dirichlet = [];
list_loadface = [];
list_noloadface = [];
I_list = zeros(144*N,1);
J_list = zeros(144*N,1);
X_list = zeros(144*N,1);
lambda=1;
mu=1;

%solutions and load functions
%clamped sphere
u_1 = @(x) 1-norm(x);
u_2 = @(x) 0;
u_3 = @(x) 0;
f_1 = @(x) 2*lambda+8*mu;
f_2 = @(x) 0;
f_3 = @(x) 0;

%Quadrature points and weights, fixed at 5 for the attenuation function.
Nq = 5;
%rho = [-4/5 9/20 9/20 9/20 9/20];

for t=1:Nk
    %Grab tetrahedron points
    pt = [p(tri(t,:),:),ones(4,1)];
    
    %solve for linear basis function coefficients
    cpt = pt\ip;
    
    %grab physical volume of tetrahedron
    vt = (1/6)*abs(det([pt(1,1:3)'-pt(4,1:3)',pt(2,1:3)'-pt(4,1:3)',pt(3,1:3)'-pt(4,1:3)']));
        
    %Build tensor of derivatives
    tdiv = build_derivative_nonlocal(cpt);
    
    %Construct node connection matrix
    pos_t= 3*(tri(t,:)-1)+1;
    %C_t is now sparse, constructed with triplets
    C_t = build_nodeconnection_nonlocal(pos_t,N);
    
    %Build contributing stiffness matrix
    M_tk = build_stiffness_nonlocal(lambda,mu,tdiv,tdiv);
    
    %integrate
    M_tk = M_tk*vt;
    
    %Add to global stiffness matrix
    %To speed up code, we will store values in triplets, and sparse them
    %together afterwards
    %M_coords = C_t'*M_tk*C_t;
    %Sparse lists, direct computation, no node connection matrix necessary
    I_coords = zeros(144,1);
    J_coords = I_coords;
    X_vals = J_coords;
    for i=1:12
        for j=1:12
            I_coords(12*(i-1)+j) = pos_t(floor((i-1)/3+1))+mod(i-1,3); %t'th-element
            J_coords(12*(i-1)+j) = pos_t(floor((j-1)/3+1))+mod(j-1,3); %k'th element
            X_vals(12*(i-1)+j) = M_tk(i,j); %ok
        end
    end
    %[I_coords,J_coords,X_vals] = find(M_coords);
    I_list(144*(t-1)+1:144*(t-1)+144) = I_coords;
    J_list(144*(t-1)+1:144*(t-1)+144) = J_coords;
    X_list(144*(t-1)+1:144*(t-1)+144) = X_vals;
    %M_global = sparse(M_global + C_t'*M_tk*C_t);
    %Build load vector volume (hard-coded Nq=5)
    B_t = build_load_volume(f_1,f_2,f_3,cpt,pt,vt);
    
    %Build load vector surface (hard-coded Nq=4)
    
    %Add to global load vector
    B_global = B_global + C_t'*B_t;
    
    %tru solution
    for i=1:4
        u_true(pos_t(i)) = 1-norm(pt(i,1:3))^2;
    end
    %Cube Boundary
    %{
    %Check if element has edges at boundary
    [ind1,~,~] = find(edge==tri(t,1));
    [ind2,~,~] = find(edge==tri(t,2));
    [ind3,~,~] = find(edge==tri(t,3));
    [ind4,~,~] = find(edge==tri(t,4));
    %We now have 4 lists of row indices, possibly empty,
    %each row index is where an edge with a node from the tetrahedron
    %has been found
    %It remains to find if there is a shared row index between
    %at least 3 of those lists.
    indlist = [ind1;ind2;ind3;ind4];
    if(isempty(indlist)==0)
        uniqueind = unique(indlist);
        countmatrix = [uniqueind,histc(indlist,uniqueind)];
        [indlength,~] = size(countmatrix);
        edgelist_t = [];
        for i=1:indlength
            if countmatrix(i,2) >= 3
                edgelist_t = [edgelist_t;countmatrix(i,1)];
            end
        end
        edgelist_length = length(edgelist_t);
        for j=1:edgelist_length
            list_dirichlet = [list_dirichlet;edgelist_t(j)];
        end
    end 
    %}
    disp(t);
end
%Parse Dirichlet Boundary (Cube)
%Ne = length(list_dirichlet);
M_global = sparse(I_list,J_list,X_list,3*N,3*N);
Ne = length(edge);
for s=1:Ne
    edgepos = 3*(edge(s,:)-1)+1;
    for i=1:3
        for j=1:3
            %Idea: set row to 0 and diagonal to 1 for m
            %and to 0 for that index of the load vector, to force that u*1=0
            M_global(edgepos(i)+(j-1),:) = 0;
            %set column to zero, since that u is supposed to be 0
            %maintaining symmetry
            M_global(:,edgepos(i)+(j-1)) = 0;
            %set diagonal to 1
            M_global(edgepos(i)+(j-1),edgepos(i)+(j-1)) = 1;
        end
        B_global(edgepos(i):edgepos(i)+2) = 0;
    end
end
U = M_global\B_global;
h = 1/N^3;
U_error = norm(U-u_true,inf)
%x_n = [x_n;log(h)];
%y_n = [y_n;log(U_error)];

% Begin iteration
itmax = 3;
%{
for it=1:itmax
    for t=1:Nk
    %Grab tetrahedron points
    pt = [p(tri(t,:),:),ones(4,1)];
    
    %solve for linear basis function coefficients
    cpt = pt\ip;
    
    %grab physical volume of tetrahedron
    vt = (1/6)*abs(det([pt(1,1:3)'-pt(4,1:3)',pt(2,1:3)'-pt(4,1:3)',pt(3,1:3)'-pt(4,1:3)']));
        
    %Build tensor of derivatives
    tdiv = build_derivative_nonlocal(cpt);
    %Construct node connection matrix
    pos_t= 3*(tri(t,:)-1)+1;
    %C_t is now sparse, constructed with triplets
    C_t = build_nodeconnection_nonlocal(pos_t,N);
    
    %Build local strain
    strain_t = zeros(3,3);
    for i=1:3
        for j=1:3
            strain_t(i,j) = tdiv(1,:,i,j)*C_t*U;
        end
    end
    end
end
%}
    
