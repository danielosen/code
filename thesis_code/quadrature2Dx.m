function [I] = quadrature2Dx(p1,p2,p3,Nq,g)
    I = 0;
    n = cross(p2-p1,p3-p1);
    area = (1/2)*dot(n,n)^(1/2);

    if (Nq == 1)
        z_q = [p1 p2 p3]*[1/3 1/3 1/3]';
        rho = 1;
        I = g(z_q)*rho;
        I = I*area;
    elseif (Nq == 3)
        z_q = [p1 p2 p3]*[1/2 1/2 0;
        1/2 0 1/2;
        0 1/2 1/2]';
        rho = 1/3;
        for (i=1:Nq)
            I = I + g(z_q(:,i))*rho;
        end
        I = I*area;
    elseif (Nq == 4)
        z_q = [p1 p2 p3]*[1/3 1/3 1/3;
        3/5 1/5 1/5;
        1/5 3/5 1/5;
        1/5 1/5 3/5]';
        rho = [-9/16 25/48 25/48 25/48];
        for(i=1:Nq)
            I= I + g(z_q(:,i))*rho(i);
        end
        I = I*area;
    end
    return 