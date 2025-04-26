function leads_info = setLeadMu(paramsdata, stepValue)
    % Site includes spin
    assign_parameters(paramsdata);
    variableList = strsplit(variableToUpdate, ','); % This splits the string into a cell array of strings
    for idx = 1:length(variableList)
        eval(sprintf('%s = %f;', variableList{idx}, stepValue));
    end
    %_______________to be replaced______________________________________________________________________________
%%% Specific to differnt attaced leads, Nx, Ny is system scale
    
    % Parameters
    Ny_Lead = 1; % Number of lattice sites in y-direction for each chain
    mu_Lead = 0; % Chemical potential
    
    % Pauli matrices and identity matrices
    sigma_x_Lead = [0, 1; 1, 0];
    sigma_y_Lead = [0, -1i; 1i, 0];
    sigma_z_Lead = [1, 0; 0, -1];
    I_spin_Lead = eye(2);
    I_chain_Lead = eye(Ny_Lead);
    
    % On-site potential matrix for a single site
    H_onsite_Lead = - mu_Lead * I_spin_Lead;
    
    % Single chain Hamiltonian along y-axis with NNN interactions, SOC
    % hopping and regural hopping
    T_nn_y_Lead = 1i * lambdaLead / 2 * sigma_y_Lead  + thopLead * I_spin_Lead; % Nearest-neighbor hopping along y
    H_y_Lead = kron(diag(ones(Ny_Lead - 1, 1), 1), T_nn_y_Lead) + kron(diag(ones(Ny_Lead - 1, 1), -1), T_nn_y_Lead') + ...
        kron(I_chain_Lead, H_onsite_Lead); % Add on-site term
    
    % X-axis hopping (nearest neighbor) and NNN interaction
    T_nn_x_Lead = 1i * lambdaLead / 2 * sigma_x_Lead + thopLead * I_spin_Lead; % Assuming uniform hopping along x
    H_nn_x01_Lead = kron(diag(ones(Ny_Lead, 1)), T_nn_x_Lead);
    thopLead_central = tLeadC*H_nn_x01_Lead;

%%% This is with spin, for eg site 1 is spin up and site2 is spin down,
%%% Specific to direct connecting to mzm
    % Define the lead structures
    lead1 = struct();
    lead1.mu = muLead1;
    lead1.t = H_nn_x01_Lead;
    lead1.epsilon0 = H_y_Lead;
    lead1.temperature = Temperature;
    lead1.lambda = 0;

    lead2 = struct();
    lead2.mu = muLead2;
    lead2.t = H_nn_x01_Lead;
    lead2.epsilon0 = H_y_Lead;
    lead2.temperature = Temperature;
    lead2.lambda = 0;

    lead3 = struct();
    lead3.mu = muLead3;
    lead3.t = H_nn_x01_Lead;
    lead3.epsilon0 = H_y_Lead;
    lead3.temperature = Temperature;
    lead3.lambda = 0;
    % 
    % lead4 = struct();
    % lead4.mu = muLead4;
    % lead4.t = H_nn_x01_Lead;
    % lead4.epsilon0 = H_y_Lead;
    % lead4.temperature = Temperature;
    % lead4.lambda = 0;

    % Assign positions to the leads
    
    %%
    %%% Positon formats is [x1,y1,flavor1;x1,y2,flavor2],flavor1 0 connect
    %%% to electron ck^\dagger d + h.c., flavor1 1 connect mzm1 and 2 to
    %%% mzm2
    % lead1.position = {[1,1,1;1,1,2]};
    % lead1.V1alpha = {[0.5*tLeadC;0.5*1i*tLeadC]};
    % lead2.position = {[Nx,1,1;Nx,1,2]};
    % lead2.V1alpha = {[0.5*tLeadC;0.5*1i*tLeadC]};
    % lead1.position = {[1,1,0]};
    % lead1.V1alpha = {[tLeadC]};
    % lead2.position = {[Nx,1,0]};
    % lead2.V1alpha = {[tLeadC]};
    % Site includes spin, x not spin, y include spin
    lead1.position = {[8,15,0],[8,16,0]};
    lead1.V1alpha = {[tLeadC],[tLeadC]};
    lead2.position = {[2,11,0],[2,12,0]};
    lead2.V1alpha = {[tLeadC],[tLeadC]};
    lead3.position = {[6,3,0],[6,4,0]};
    lead3.V1alpha = {[tLeadC],[tLeadC]};
%    lead3.position = [3*Nx+1];
%    lead4.position = [1];

    % Return the leads_info array
%    leads_info = {lead1, lead2, lead3, lead4};
    leads_info = {lead1,lead2,lead3};
end
