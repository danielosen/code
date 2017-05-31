%Build load vector from volume integrals (5 quadrature points hard-coded)
function [B_t] = build_load_volume(f_1,f_2,f_3,cpt,pt,vt),
B_t = zeros(12,1);
z_q = [pt(1,1:3)' pt(2,1:3)' pt(3,1:3)' pt(4,1:3)']*[1/4 1/4 1/4 1/4;
        1/2 1/6 1/6 1/6;
        1/6 1/2 1/6 1/6;
        1/6 1/6 1/2 1/6;
        1/6 1/6 1/6 1/2]';
rho = [-4/5 9/20 9/20 9/20 9/20];
    for j=1:4
        I_1 = 0;
        I_2 = 0;
        I_3 = 0;
        for i=1:5
            I_1 = I_1 + rho(i)*f_1(z_q(:,i))*cpt(:,j)'*[z_q(:,i);1];
            I_2 = I_2 + rho(i)*f_2(z_q(:,i))*cpt(:,j)'*[z_q(:,i);1];
            I_3 = I_3 + rho(i)*f_3(z_q(:,i))*cpt(:,j)'*[z_q(:,i);1];
        end
        B_t(3*(j-1)+1) = I_1*vt;
        B_t(3*(j-1)+2) = I_2*vt;
        B_t(3*(j-1)+3) = I_3*vt;  
    end
end