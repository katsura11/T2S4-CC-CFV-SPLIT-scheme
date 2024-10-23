function  C_ext = cal_exact_sol(N, Nt, T_max)

Xmin = -4; Xmax = 4;
Ymin = -4; Ymax = 4;

% 后续计算
dt = T_max/Nt;
h = (Xmax - Xmin) / N;


%% 
    C = zeros(3,3);
    coef = [1/6, 4/6, 1/6];
    t = Nt;
    for i = 1:N     
        for j = 1:N
            for ii = 0:2
                x = Xmin + (i-1+0.5*ii)*h;
                for jj = 0:2
                    y = Ymin + (j-1+0.5*jj)*h;
                    r = sqrt(x^2+y^2);
                    f_t = tanh(r)./cosh(r)^2;
                    C(ii+1, jj+1) = -tanh(y/2.*cos(f_t*dt*t/(0.385*r)) - x/2.*sin(f_t*dt*t/(0.385*r)));
                    if r == 0
                        C(ii+1, jj+1) = 0;
                        %fprintf("ii = %3d, jj = %3d, r = %f, f_t = %f, C = %f\n", ii, jj, r, f_t, C(ii+1, jj+1));
                    end
                end
            end           
            C_ext(j,i) = coef*C*coef';
        end
    end

end