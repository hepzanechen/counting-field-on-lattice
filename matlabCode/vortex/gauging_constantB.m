function H_gauged = gauging_constantB(H, Lx, Ly, B, orbPerSite)
    % gauging_constantB applies a uniform magnetic field B using the gauge A_x = By
    % H: Hamiltonian matrix (BdG form)
    % Lx, Ly: dimensions of the system
    % B: magnetic field strength
    % orbPerSite: An integer
    % Define constants
    [num_sites, ~] = size(H);
    flux_quantum = 1; % Assuming e/h = 1 for simplicity; adjust as necessary

    % Check for dimension mismatch

    if 2*orbPerSite * Lx * Ly ~= num_sites
        warning('Dimension mismatch: 4 * Lx * Ly is not equal to the number of sites.');
    end

    % Initialize the gauged Hamiltonian
    H_gauged = H;

    % Define neighbors and their direction vectors (same as original)
    neighbors_directions = {[1, 0],  [-1, 0], [0, 1],  [0, -1], [2, 0],  [-2, 0], [0, 2],  [0, -2]};

    % Loop through each site
    for x = 1:Lx
        for y = 1:Ly
            % Loop through each neighbor direction
            for k = 1:length(neighbors_directions)
                dr = neighbors_directions{k};
                r_n = [x, y] + dr;

                % Check if the neighbor is within bounds, rigid boundary conditions
                hopping_entry_postion_shift = dr * [Ly, 1]';
                hopping_entry_postion = [(x - 1) * Ly + y, (x - 1) * Ly + y + hopping_entry_postion_shift];  
                % Make sure the rigid boundary conditions
                new_position = [x, y] + dr;
                if new_position(1) >= 1 && new_position(1) <= Lx && ...
                   new_position(2) >= 1 && new_position(2) <= Ly
                    % This gauge can be changed
                    % B0gauge1
                    A_gauge=[B*(y),0];
                    %B0gauge20
                    % A_gauge=[B*y-0.3,0];
                    % A_gauge=[B*(y-1),0];
                    % A_gauge=[0,0];
                    % Phase=A.dr
                    phase = 2 * pi * A_gauge * dr' / flux_quantum;


                    ele_hopping_entry_inHMatrix_x = 2*orbPerSite * hopping_entry_postion(1) - 2*(orbPerSite-1)-1:2:2*orbPerSite * hopping_entry_postion(1)-1;
                    ele_hopping_entry_inHMatrix_y = 2*orbPerSite * hopping_entry_postion(2) - 2*(orbPerSite-1)-1:2:2*orbPerSite * hopping_entry_postion(2)-1;
                    hole_hopping_entry_inHMatrix_x = 2*orbPerSite * hopping_entry_postion(1) - 2*(orbPerSite-1):2:2*orbPerSite * hopping_entry_postion(1);
                    hole_hopping_entry_inHMatrix_y = 2*orbPerSite * hopping_entry_postion(2) - 2*(orbPerSite-1):2:2*orbPerSite * hopping_entry_postion(2);


                    % Apply phase to electron and hole parts
                    H_gauged(ele_hopping_entry_inHMatrix_x, ele_hopping_entry_inHMatrix_y) = ...
                        H(ele_hopping_entry_inHMatrix_x, ele_hopping_entry_inHMatrix_y) * exp(1i * phase);
                    H_gauged(hole_hopping_entry_inHMatrix_x, hole_hopping_entry_inHMatrix_y) = ...
                        H(hole_hopping_entry_inHMatrix_x, hole_hopping_entry_inHMatrix_y) * exp(-1i * phase);
                end
            end
        end
    end
end
