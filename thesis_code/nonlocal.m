%Non-Local Finite Element Method Script
%For the linear elastic element
%Author: Daniel Osen 2016

%Choose Computational Domain
%Sphere Case, N = number of nodes
%N = 189;
%[p,tri,edge] = getSphere(N);

%Cube case, N = number of nodes (afterwards)
N = 8^4;
[p,tri,edge] = getCube(N);
[N,~] = size(p);

%Grab number of elements Nk, NP=4
[Nk,Np] = size(tri);
[N_edge,Np_edge] = size(edge);
%Pre-Allocate matrices
ip = eye(4);
id = eye(3);
M_global = zeros(3*N,3*N);
B_global = zeros(3*N,1);
U_classical = zeros(3*N,1);
list_dirichlet = [];
list_loadface = [];
list_noloadface = [];

%Attenuation Function (Euclidean Distance)
%material length scale l>0, 
atn_l = 0.00001;%
%Finite influence distance parameter
atn_alpha = 1;
atn_r = atn_l*(1+atn_alpha);
%Normalization constant (Sphere)
atn_k = 1/(atn_l^3*(pi)^(3/2));
%function call
atn = @(x,y) atn_k*exp(-(norm(y-x))^2/atn_l^2);

%Lame constants
lambda=2;
mu=2;

%solutions and load functions
%clamped sphere
%u_1 = @(x) 1-norm(x);
%u_2 = @(x) 0;
%u_3 = @(x) 0;
%f_1 = @(x) 2*lambda+8*mu;
%f_2 = @(x) 0;
%f_3 = @(x) 0;
%cube clamped at one end
%f_surf_1 = @(x) 0;
%f_surf_2 = @(x) 0;
%f_surf_3 = @(x) -(x(1)^2-x(1))^2*(x(2)^2-x(2))^2;
%f_1 = @(x) 0;
%f_2 = @(x) 0;
%f_3 = @(x) -x(3)*(x(1)^2-x(1))^2*(x(2)^2-x(2))^2;
%cube clamped at all ends
f_1 = @(x) (-1)*((lambda+2*mu)*2*(x(2)^2-x(2))*(x(3)^2-x(3))+mu*2*(x(1)^2-x(1))*(x(3)^2-x(3))+mu*2*(x(1)^2-x(1))*(x(3)^2-x(3))+(lambda+mu)*(2*x(1)-1)*(2*x(2)-1)*(x(3)^2-x(3))+(lambda+mu)*(2*x(1)-1)*(2*x(3)-1)*(x(2)^2-x(2)));
f_2 = @(x) (-1)*((lambda+2*mu)*2*(x(1)^2-x(1))*(x(3)^2-x(3))+mu*2*(x(2)^2-x(2))*(x(3)^2-x(3))+mu*2*(x(2)^2-x(2))*(x(3)^2-x(3))+(lambda+mu)*(2*x(2)-1)*(2*x(1)-1)*(x(3)^2-x(3))+(lambda+mu)*(2*x(2)-1)*(2*x(3)-1)*(x(1)^2-x(1)));
f_3 = @(x) (-1)*((lambda+2*mu)*2*(x(2)^2-x(2))*(x(1)^2-x(3))+mu*2*(x(3)^2-x(3))*(x(1)^2-x(3))+mu*2*(x(3)^2-x(3))*(x(1)^2-x(1))+(lambda+mu)*(2*x(3)-1)*(2*x(2)-1)*(x(1)^2-x(1))+(lambda+mu)*(2*x(3)-1)*(2*x(1)-1)*(x(2)^2-x(2)));
f_surf_1 = @(x) 0;
f_surf_2 = @(x) 0;
f_surf_3 = @(x) 0;
u_true = zeros(3*N,1);
u_cube = @(x) (x(1)^2-x(1))*(x(2)^2-x(2))*(x(3)^2-x(3));



%Quadrature points and weights, fixed at 5 for the attenuation function.
Nq = 5;
rho = [-4/5 9/20 9/20 9/20 9/20];

%Begin iteration over elements
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
    C_t = build_nodeconnection_nonlocal(pos_t,N);
    
    %Get quadrature points
    z_qt = [pt(1,1:3)',pt(2,1:3)',pt(3,1:3)',pt(4,1:3)']*[1/4 1/4 1/4 1/4;1/2 1/6 1/6 1/6;1/6 1/2 1/6 1/6;1/6 1/6 1/2 1/6;1/6 1/6 1/6 1/2]';
    
    %Local Solution
    u_true(pos_t) = u_cube(pt(:,1:3));
    u_true(pos_t+1) = u_true(pos_t);
    u_true(pos_t+2) = u_true(pos_t);
    
    %Inner element Loop
    for k=1:Nk
        
        %Grab tetrahedron points
        pk = [p(tri(k,:),:),ones(4,1)];
        
        %If the distance between the elements is large enough,
        %it is computationally efficient to ignore any contributions,
        %at presumably a small loss of error with the correct choice of the
        %finite influence distance R. This distance is only considered
        %large or small relative to the material length scale l. Hence if
        %r/l >> 1, then atn is considered 0, so we set R = l(1+alpha), where
        %alpha is some positive number, possibly small.
        dist_center = sum(pk(:,1:3))/4-sum(pt(:,1:3))/4;
        if norm(dist_center)<inf
    
        %solve for linear basis function coefficients
        cpk = pk\ip;
    
        %grab physical volume of tetrahedron
        vk = (1/6)*abs(det([pk(1,1:3)'-pk(4,1:3)',pk(2,1:3)'-pk(4,1:3)',pk(3,1:3)'-pk(4,1:3)']));
        
        %Build tensor of derivatives
        kdiv = build_derivative_nonlocal(cpk);
        
        %Build contributing stiffness matrix
        M_tk = build_stiffness_nonlocal(lambda,mu,tdiv,kdiv);
        
        %Integrate stiffness matrix
        %Apart from the attenuation function, everything beneath the
        %integral sign, as described in the thesis, is constant.
        %This allows us to take the stiffness matrix out of the integral.
        %We integrate the attenuation function using the quadrature rules
        %employed in the classical case. Here we limit the quadrature rule
        %to 5 points.
        int_atn = 0;
        %get quadrature points
        z_qk = [pk(1,1:3)',pk(2,1:3)',pk(3,1:3)',pk(4,1:3)']*[1/4 1/4 1/4 1/4;1/2 1/6 1/6 1/6;1/6 1/2 1/6 1/6;1/6 1/6 1/2 1/6;1/6 1/6 1/6 1/2]';        
        for i=1:5
            for j=1:5
                int_atn = int_atn + rho(i)*rho(j)*atn(z_qt(:,i),z_qk(:,j));
            end
        end
        %scale with volume and make to local stiffness matrix
        int_atn = int_atn*vt*vk;
        M_tk = int_atn*M_tk;
        
        %Build node connection matrix
        pos_k= 3*(tri(k,:)-1)+1;
        C_k = build_nodeconnection_nonlocal(pos_k,N);
        
        %Add to global matrix
        M_global = M_global + C_t'*M_tk*C_k;
        
        else
        end
                
    end
    
    %Build load vector.
    B_t = build_load_volume(f_1,f_2,f_3,cpt,pt,vt);
    %Add to global load vector
    B_global = B_global + C_t'*B_t;
    
    %Cube Boundary
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
        [indlength,notused] = size(countmatrix);
        edgelist_t = [];
        for i=1:indlength
            if countmatrix(i,2) >= 3
                edgelist_t = [edgelist_t;countmatrix(i,1)];
            end
        end
        edgelist_length = length(edgelist_t);
        %edgelist_t now contains all the row indices of edge
        %which has at exactly three nodes shared with the nodes in the
        %tetrahedron
        for j=1:edgelist_length
            %Cube/sphere case clamped at all ends (only dirichlet boundary)
            list_dirichlet = [list_dirichlet;edgelist_t(j)];
            %Cube case clamped at one end
            %{
            p_edge_t = p(edge(edgelist_t(j),:),:);
            %If y is 0, we are the clamped boundary and add this to be parsed
            %later
            
            if abs(p_edge_t(1,2))<eps && abs(p_edge_t(2,2))<eps && abs(p_edge_t(3,2))<eps
                list_dirichlet = [list_dirichlet;edgelist_t(j)];
            %if z=1 we are being pushed by a force and need to integrate
            elseif abs(1-p_edge_t(1,3))<eps && abs(1-p_edge_t(2,3))<eps && abs(1-p_edge_t(3,3))<eps
                B_surf_t = zeros(12,1);
                list_loadface = [list_loadface;edgelist_t(j)];
                for i=1:4
                    %NB: Anonymous functions in matlab are notoriously slow but...
                    %multiply the vector load function with the vector shape function
                    %done component wise for each node
                    fshape_1 = @(x) f_surf_1(x)*[x(1),x(2),x(3),1]*cpt(:,i);
                    fshape_2 = @(x) f_surf_2(x)*[x(1),x(2),x(3),1]*cpt(:,i); 
                    fshape_3 = @(x) f_surf_3(x)*[x(1),x(2),x(3),1]*cpt(:,i); 
                    %compute the integral for each component using quadrature rule
                    %using running index, this is done in physical space
                    B_surf_t(3*(i-1)+1) = quadrature2Dx(p_edge_t(1,1:3)',p_edge_t(2,1:3)',p_edge_t(3,1:3)',4,fshape_1);
                    B_surf_t(3*(i-1)+2) = quadrature2Dx(p_edge_t(1,1:3)',p_edge_t(2,1:3)',p_edge_t(3,1:3)',4,fshape_2);
                    B_surf_t(3*(i-1)+3) = quadrature2Dx(p_edge_t(1,1:3)',p_edge_t(2,1:3)',p_edge_t(3,1:3)',4,fshape_3); 
                end
                B_global = B_global + C_t'*B_surf_t;
            else
                %we are neither at the clamped boundary nor at the boundary
                %being pushed, in this case the forces are zero, hence the
                %integral is zero, and we don't have to do anything.
                list_noloadface = [list_noloadface;edgelist_t(j)];
            end
            %}
            
        end
    else
    end
            
end

%Parse Dirichlet Boundary (Cube)
Ne = length(list_dirichlet);
for s=1:Ne
    edgepos = 3*(edge(list_dirichlet(s),:)-1)+1;
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
%{
%Parse Dirchlet Boundary (Clamped Sphere)
[Ne, Npe] = size(edge);
for k=1:Ne
    edgepos = 3*(edge(k,:)-1)+1; %grab running index
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
%}
%Solve using backslash
U_global = M_global\B_global;
quiver3(p(:,1),p(:,2),p(:,3),U_global(1:3:end),U_global(2:3:end),U_global(3:3:end))
%spy(M_global)