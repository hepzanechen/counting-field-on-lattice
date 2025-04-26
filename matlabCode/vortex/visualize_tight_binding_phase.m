function visualize_tight_binding_phase(H_total, Nx, Ny,m)
    % Initialize matrices to store the vector field for phases
    phase_x_particle = zeros(Nx, Ny);
    phase_y_particle = zeros(Nx, Ny);
    phase_x_hole = zeros(Nx, Ny);
    phase_y_hole = zeros(Nx, Ny);

    % the background phase is determined carefully from tight-binding
    % Hamiltonian
    ele_backgroundPhase=angle(-m);
    hole_backgroundPhase=angle(m);
    % ele_backgroundPhase=0;
    % hole_backgroundPhase=0;
    % Loop over all lattice sites
    for ix = 1:Nx
        for iy = 1:Ny
            % Calculate the index in the Hamiltonian matrix
            % For each lattice point, considering BdG form and spin
            index_base = ((ix - 1) * Ny + iy - 1) * 4 + 1;
            
            % Hopping in x-direction (to the right)
            if ix < Nx
                index_right = ((ix) * Ny + iy - 1) * 4 + 1;
                % Extract particle-particle sector (H_11)
                hopping_x_particle = H_total(index_base, index_right);
                % Extract hole-hole sector (H_22)
                hopping_x_hole = H_total(index_base + 1, index_right + 1);
                % Calculate phases
                phase_x_particle(ix, iy) = angle(hopping_x_particle)-ele_backgroundPhase;
                phase_x_hole(ix, iy) = angle(hopping_x_hole)-hole_backgroundPhase;
            end
            
            % Hopping in y-direction (upwards)
            if iy < Ny
                index_up = ((ix - 1) * Ny + iy) * 4 + 1;
                % Extract particle-particle sector (H_11)
                hopping_y_particle = H_total(index_base, index_up);
                % Extract hole-hole sector (H_22)
                hopping_y_hole = H_total(index_base + 1, index_up + 1);
                % Calculate phases
                phase_y_particle(ix, iy) = angle(hopping_y_particle)-ele_backgroundPhase;
                phase_y_hole(ix, iy) = angle(hopping_y_hole)-hole_backgroundPhase;
            end
        end
    end
    
    % Plot the phase field for particle sector
    figure;
    quiver(1:Nx, 1:Ny, phase_x_particle, phase_y_particle, 'AutoScaleFactor', 0.5);
    axis equal;
    xlabel('x');
    ylabel('y');
    title('Phase Field of Tight-Binding Model (Particle Sector)');
    colorbar;
    
    % Plot the phase field for hole sector
    figure;
    quiver(1:Nx, 1:Ny, phase_x_hole, phase_y_hole, 'AutoScaleFactor', 0.5);
    axis equal;
    xlabel('x');
    ylabel('y');
    title('Phase Field of Tight-Binding Model (Hole Sector)');
    colorbar;
end
