% Pennes Bioheat Equation in 1D Spherical Coordinates using Matrix Inverse
clear; clc;

%% === Model Parameters ===
R = 5e-3;         % Radius [m]
dr = 0.1e-3;      % Spatial step [m]
dt = 1;           % Time step [s]
t_end = 600;      % Total time [s]

r = (0:dr:R)';    % Radial grid
N = length(r);
time = 0:dt:t_end;
Nt = length(time);

% Tissue properties
rho = 1050; c = 3500; k = 0.5; w_b = 0.00088;
rho_b = 1050; c_b = 3617; T_b = 37;
Q_met = 5790; Q_ext = 1e5;

% Boundary and initial conditions
T_0 = 37; T_inf = 23; h = 10;

% Precomputed coefficients
theta = rho * c / dt;
beta = k / dr^2;
lambda = rho_b * c_b * w_b;
gamma = k ./ (r(2:end-1) * dr);

%% === Initialize Temperature and Matrix ===
T = zeros(N, Nt);
T(:,1) = T_0;

A = zeros(N, N);  % Coefficient matrix

% Interior nodes
for i = 2:N-1
    gi = gamma(i-1);
    A(i,i-1) = -(beta - gi);
    A(i,i)   = theta + 2*beta + lambda;
    A(i,i+1) = -(beta + gi);
end

% Center node (symmetry)
A(1,1) = theta + 2*beta + lambda;
A(1,2) = -2*beta;

% Outer node (convection)
A(N,N-1) = -k;
A(N,N)   = k + h * dr;


%% === Time-Stepping Loop with Richardson Iteration ===
for n = 1:Nt-1
    b = zeros(N, 1);
    b(1) = theta * T(1,n) + lambda * T_b + Q_met + Q_ext;
    for i = 2:N-1
        b(i) = theta * T(i,n) + lambda * T_b + Q_met + Q_ext;
    end
    b(N) = h * dr * T_inf;

    % Solve linear system
    T(:,n+1) = A \ b;
end


%% === Rescaling Matrix and Right-hand Side for Richardson Iterations===
scale_A = max(abs(diag(A)));
A_scaled = A / scale_A;

% Richardson parameters
tol = 1e-12;
max_iter = 5000;
omega = 0.9*(1 / max(diag(A_scaled)));  % safe choice after scaling

%% === Time-Stepping Loop with Richardson Iteration ===
TR = zeros(N, Nt);
TR(:,1) = T_0;
for n = 1:Nt-1
    b = zeros(N,1);
    b(1) = theta * TR(1,n) + lambda * T_b + Q_met + Q_ext;
    for i = 2:N-1
        b(i) = theta * TR(i,n) + lambda * T_b + Q_met + Q_ext;
    end
    b(N) = h * dr * T_inf;

    % Rescale b
    b_scaled = b / scale_A;

    % Initial guess
    T0 = TR(:,n);

    % Richardson Iteration
    [T_new, iter_used] = richardson_solver(A_scaled, b_scaled, T0, omega, tol);
    TR(:,n+1) = T_new;

    % Optional: print iteration count
    fprintf('Time %4ds: Richardson converged in %d iterations\n', time(n+1), iter_used);
end
%% === Plot Results ===
close all
figure;

% Subplot 1: Final profile
subplot(2,1,1);
plot(r*1000, T(:,end), 'r-', 'LineWidth', 2); hold on
plot(r*1000, TR(:,end), 'r--', 'LineWidth', 2);
xlabel('Radial Position (mm)');
ylabel('Temperature (°C)');
title('R Final Temperature Distribution (t = 600 s)');
legend('Inverser', 'Rich.Iter')
grid on;

% Subplot 2: T_center and T_surface vs time
subplot(2,1,2);
plot(time, T(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(time, T(end,:), 'k-', 'LineWidth', 1.5);
plot(time, TR(1,:), 'b--', 'LineWidth', 1.5); 
plot(time, TR(end,:), 'k--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Temperature (°C)');
legend('Center (r = 0)', 'Surface (r = R)', 'Location', 'best');
title('Temperature Evolution at Center and Surface');
grid on;

%% === Richardson Solver Function ===
function [x, iter] = richardson_solver(A, b, x0, omega, tol)
    x = x0;
    iter = 0;
    r = ones(1,length(x0));
    while norm(r, inf) >= tol
        r = b - A * x;
        x_new = x + omega * r;
        x = x_new;
        iter = iter+1;
        if iter > 1e6
            disp('Richardson method did not converge within 1e6 iterations.');
            break;
        end
    end
end