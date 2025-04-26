function H_total = construct_Hamiltonian(paramsdata, stepValue)
    %%%
    assign_parameters(paramsdata);
    variableList = strsplit(variableToUpdate, ','); % This splits the string into a cell array of strings
    for idx = 1:length(variableList)
        eval(sprintf('%s = %f;', variableList{idx}, stepValue));
    end
   %_______________to be replaced______________________________________________________________________________
    Q=2*pi;
    vF=1;
    H_p = construct_Hamiltonian_p(vF, Q)
    H_m = construct_Hamiltonian_m(vF, Q)
    H_BdG=kron(H_p,[1,0;0,0])+H_m,[0,0;0,-1]);
    
    H_Delta_spin_cell = [
        0,          0,          0,          Delta;
        0,          0,          Delta,      0;
        0,          Delta,      0,          0;
        Delta,      0,          0,          0
    ];
    H_Delta =kron(eye(Nx*Ny),H_Delta_spin_cell);
    H=H_BdG+H_Delta;


    % H_total=gauging(H, Nx, Ny, lambda_s, vortices,ifSpin);
    H_total=gaugingWithPlot(H, Nx, Ny, lambda_s, vortices,ifSpin);
    % H_total=gauging_constantB(H, Nx, Ny, B, orbPerSite);
end
function H_p = construct_Hamiltonian_p(vF, Q)
    % Pauli matrices and identity matrices
    sigma_x = [0, 1; 1, 0];
    sigma_y = [0, -1i; 1i, 0];
    sigma_z = [1, 0; 0, -1];
    I_spin = eye(2);
    I_chain = eye(Ny);
    I_Nx = eye(Nx);
    
    % On-site potential matrix for a single site
    H_onsite = -vF^2*(1/8)*1i*Q^2*sigma_x;
    
    % Single chain Hamiltonian along y-axis with NNN interactions
    T_nn_y = vF^2*sigma_x; % Nearest-neighbor hopping along y
    H_y = kron(diag(ones(Ny-1, 1),1), T_nn_y) + kron(diag(ones(Ny-1, 1), -1), T_nn_y') + ...
         kron(I_chain, H_onsite); % Add on-site term
    
    
    % X-axis hopping (nearest neighbor) and NNN interaction
    T_nn_x = -vF^2*sigma_x; % Assuming uniform hopping along x
    H_nn_x01 = kron(diag(ones(Ny, 1)) , T_nn_x);
    
    % NNN interaction ,diagnoal hopping
    T_nnn_x01y01 =  -(1i/2)*vF^2*sigma_x; %\hat{c}_{x-1,y-1,A}^{\dagger}\hat{c}_{x,y,B}
    T_nnn_x01y10 =  (1i/2)*vF^2*sigma_x; %\hat{c}_{x+1,y-1,A}^{\dagger}\hat{c}_{x,y,B}
    H_nnn_x01 = kron(diag(ones(Ny-1, 1),1) , T_nnn_x01y01)+kron(diag(ones(Ny-1, 1),-1) , T_nnn_x01y10);
    
    % Assemble full 2D Hamiltonian with NNN
    H_p = kron(I_Nx, H_y) + ... % Diagonal blocks (chains along y)
              kron(diag(ones(Nx-1, 1), 1), H_nn_x01) + ... % Right off-diagonal blocks (hopping to right chain)
              kron(diag(ones(Nx-1, 1), -1), H_nn_x01' )+...; % Left off-diagonal blocks (Hermitian conjugate)
              kron(diag(ones(Nx-2, 1), 2), H_nnn_x01) + ... % Next Right off-diagonal blocks (hopping to next right chain)
              kron(diag(ones(Nx-2, 1), -2), H_nnn_x01' ); % Left off-diagonal blocks (Hermitian conjugate)

end
function H_m = construct_Hamiltonian_m(vF, Q)
    % Pauli matrices and identity matrices
    sigma_x = [0, 1; 1, 0];
    sigma_y = [0, -1i; 1i, 0];
    sigma_z = [1, 0; 0, -1];
    I_spin = eye(2);
    I_chain = eye(Ny);
    I_Nx = eye(Nx);
    
    % On-site potential matrix for a single site
    H_onsite = vF^2*(1/8)*1i*Q^2*sigma_x;
    
    % Single chain Hamiltonian along y-axis with NNN interactions
    T_nn_y = vF^2*sigma_x; % Nearest-neighbor hopping along y
    H_y = kron(diag(ones(Ny-1, 1),1), T_nn_y) + kron(diag(ones(Ny-1, 1), -1), T_nn_y') + ...
         kron(I_chain, H_onsite); % Add on-site term
    
    
    % X-axis hopping (nearest neighbor) and NNN interaction
    T_nn_x = -vF^2*sigma_x; % Assuming uniform hopping along x
    H_nn_x01 = kron(diag(ones(Ny, 1)) , T_nn_x);
    
    % NNN interaction ,diagnoal hopping
    T_nnn_x01y01 =  (1i/2)*vF^2*sigma_x; %\hat{c}_{x-1,y-1,A}^{\dagger}\hat{c}_{x,y,B}
    T_nnn_x01y10 =  -(1i/2)*vF^2*sigma_x; %\hat{c}_{x+1,y-1,A}^{\dagger}\hat{c}_{x,y,B}
    H_nnn_x01 = kron(diag(ones(Ny-1, 1),1) , T_nnn_x01y01)+kron(diag(ones(Ny-1, 1),-1) , T_nnn_x01y10);
    
    % Assemble full 2D Hamiltonian with NNN
    H_m = kron(I_Nx, H_y) + ... % Diagonal blocks (chains along y)
              kron(diag(ones(Nx-1, 1), 1), H_nn_x01) + ... % Right off-diagonal blocks (hopping to right chain)
              kron(diag(ones(Nx-1, 1), -1), H_nn_x01' )+...; % Left off-diagonal blocks (Hermitian conjugate)
              kron(diag(ones(Nx-2, 1), 2), H_nnn_x01) + ... % Next Right off-diagonal blocks (hopping to next right chain)
              kron(diag(ones(Nx-2, 1), -2), H_nnn_x01' ); % Left off-diagonal blocks (Hermitian conjugate)

end