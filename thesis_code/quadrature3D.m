function [I] = quadrature3D(p1,p2,p3,p4,Nq,g)
    alpha = 1/4 + 3*sqrt(5)/20;
    beta = 1/4 - sqrt(5)/20;
    I = 0;
    volume = (1/6)*abs(det([p1-p4,p2-p4,p3-p4]));
    if (Nq == 1)
        z_q = [p1 p2 p3 p4]*[1/4 1/4 1/4 1/4]';
        rho = 1;
        I = g(z_q)*rho;
        I = I*volume;
    elseif (Nq == 4)
        z_q = [p1 p2 p3 p4]*[alpha beta beta beta;
        beta alpha beta beta;
        beta beta alpha beta;
        beta beta beta alpha]';
        rho = 1/4;
        for (i=1:Nq)
            I = I + rho*g(z_q(:,i));
        end
        I = I*volume;
    elseif (Nq == 5)
        z_q = [p1 p2 p3 p4]*[1/4 1/4 1/4 1/4;
        1/2 1/6 1/6 1/6;
        1/6 1/2 1/6 1/6;
        1/6 1/6 1/2 1/6;
        1/6 1/6 1/6 1/2]';
        rho = [-4/5 9/20 9/20 9/20 9/20];
        for (i=1:Nq)
            I = I + rho(i)*g(z_q(:,i));
        end
        I = I*volume;
    end
    return