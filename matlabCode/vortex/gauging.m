function H_gauged = gauging(H, Lx, Ly, lambda_s, vortices,ifspin)
    % gauging applies a gauge transformation to the Hamiltonian matrix H
    % H: Hamiltonian matrix(BdG form)
    % Lx, Ly: dimensions of the system
    % lambda_s: parameter for gauge transformation
    % vortices: struct with fields 'A' and 'B', each containing positions
    % and vortex flux

    % Define constants
    [num_sites, ~] = size(H);
    
    % Add warning if 2*Lx*Ly is not equal to num_sites
    if ifspin
        if 4 * Lx * Ly ~= num_sites
            warning('Dimension mismatch: 2 * Lx * Ly is not equal to the number of sites.');
        end
    else
        if 2 * Lx * Ly ~= num_sites
            warning('Dimension mismatch: 2 * Lx * Ly is not equal to the number of sites.');
        end


    end

    % Define grid vectors
    [GX, GY] = meshgrid(2*pi*(0:(Lx-1))/Lx, 2*pi*(0:(Ly-1))/Ly);
    G = [GX(:), GY(:)];
    G_cross = [GY(:), -GX(:)];  % Cross product with z
    

    % Initialize the gauged Hamiltonian
    H_gauged = H;

    % Define neighbors and their direction vectors
    neighbors_directions = {[1, 0],[-1, 0],[0, 1],[0, -1],[2, 0],[-2, 0],[0, 2],[0, -2]};

    % Loop through each site
    for x = 1:Lx
        for y = 1:Ly
            r_m = [x, y];

            % Loop through each neighbor direction
            for k = 1:length(neighbors_directions)
                dr = neighbors_directions{k};
                r_n = r_m + dr;

                % Check if the neighbor is within bounds, rigid boundary
                % conditons
                hopping_entry_postion_shift = dr*[Ly,1]';
                if hopping_entry_postion_shift > 0    % H01 
                    hopping_entry_postion=[(x-1)*Ly + y, (x-1)*Ly + y + hopping_entry_postion_shift];
                else                                  % H10 
                    hopping_entry_postion=[(x-1)*Ly + y - hopping_entry_postion_shift, (x-1)*Ly + y];
                end    
                %  Make sure the rigid boundary condtions
                if hopping_entry_postion(1) >= 1 && hopping_entry_postion(1) <= Lx*Ly...
                && hopping_entry_postion(2) >= 1 && hopping_entry_postion(2) <= Lx*Ly

                    % Initialize phases
                    phase_A = 0;
                    phase_B = 0;

                    % Loop through vortices of type A
                    for j = 1:size(vortices.A, 1)
                        r_g = vortices.A(j, 1:2);
                        V_gFlux = vortices.A(j, 3);
                        Vgalpha = V_gFlux * 1i * (2*pi/(Lx*Ly)) * sum(G_cross * dr'...
                             .* exp(1i * G * (r_m  - r_g)') ./(lambda_s^-2 + sum(G.^2, 2)),1);                        
                        phase_A = phase_A + real(Vgalpha);
                    end

                    % Loop through vortices of type B
                    for j = 1:size(vortices.B, 1)
                        r_g = vortices.B(j, 1:2);
                        V_gFlux = vortices.B(j, 3);
                        Vgalpha = V_gFlux * 1i * (2*pi/(Lx*Ly)) * sum(G_cross * dr'...
                             .* exp(1i * G * (r_m - r_g)') ./(lambda_s^-2 + sum(G.^2, 2)),1);                        
                        phase_B = phase_B + real(Vgalpha);
                    end
                               
                    if  ifspin
                        ele_hopping_entry_inHMatrix_x = [4*hopping_entry_postion(1)-3,4*hopping_entry_postion(1)-1];
                        ele_hopping_entry_inHMatrix_y = [4*hopping_entry_postion(2)-3,4*hopping_entry_postion(2)-1];
                        hole_hopping_entry_inHMatrix_x = [4*hopping_entry_postion(1)-2,4*hopping_entry_postion(1)];
                        hole_hopping_entry_inHMatrix_y = [4*hopping_entry_postion(2)-2,4*hopping_entry_postion(2)];

                    else
                        ele_hopping_entry_inHMatrix_x = 2*hopping_entry_postion(1)-1;
                        ele_hopping_entry_inHMatrix_y = 2*hopping_entry_postion(2)-1;
                        hole_hopping_entry_inHMatrix_x = 2*hopping_entry_postion(1);
                        hole_hopping_entry_inHMatrix_y = 2*hopping_entry_postion(2);

                    end
                    % Apply electron part (phase_A)
                    H_gauged(ele_hopping_entry_inHMatrix_x, ele_hopping_entry_inHMatrix_y) = H(ele_hopping_entry_inHMatrix_x, ele_hopping_entry_inHMatrix_y) * exp(1i * phase_A);

                    % Apply hole part (phase_B)
                    H_gauged(hole_hopping_entry_inHMatrix_x, hole_hopping_entry_inHMatrix_y) = H(hole_hopping_entry_inHMatrix_x, hole_hopping_entry_inHMatrix_y) * exp(-1i * phase_B);
                end
            end
        end
    end

    
end

