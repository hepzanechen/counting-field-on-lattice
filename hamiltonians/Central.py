import torch
from typing import List, Tuple
import random

class Central:
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor):
        """
        Initializes the base central Hamiltonian with roles of x and y switched.

        Parameters:
        -----------
        Ny : int
            Number of lattice sites in the y-direction.
        Nx : int
            Number of lattice sites in the x-direction.
        t_y : torch.Tensor
            Hopping parameter in the y-direction.
        t_x : torch.Tensor
            Hopping parameter in the x-direction.
        """
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.funcDevice=t_x.device

        # Construct the Hamiltonian matrices
        self.H_chain_y = self._construct_chain_y()
        #  TODO: make * to kron when expand t_x, now it seems only support scalr t_x
        self.H_along_x = self.t_x * torch.eye(Ny, dtype=torch.complex64,device=self.funcDevice)

        # Assemble the full Hamiltonian
        self.H_full = self._assemble_full_hamiltonian()

    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along the y-direction."""
        #  TODO: make * to kron when expand t_y
        H_inter_y = self.t_y * torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64,device=self.funcDevice), 1)
        H_chain_y = H_inter_y + H_inter_y.T.conj()
        return H_chain_y

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        """Assembles the full Hamiltonian matrix without disorder effects."""
        H_full_diag = torch.kron(torch.eye(self.Nx, dtype=torch.complex64,device=self.funcDevice), self.H_chain_y)
        H_full_diag1 = torch.kron(torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64,device=self.funcDevice), 1), self.H_along_x)
        return H_full_diag + H_full_diag1 + H_full_diag1.T.conj()

    def __repr__(self):
        return f"Central(Ny={self.Ny}, Nx={self.Nx}, t_y={self.t_y}, t_x={self.t_x})"



class DisorderedCentral(Central):
    def __init__(self, Nx: int, Ny: int, t_x: torch.Tensor, t_y: torch.Tensor, deltaV: torch.Tensor, N_imp: int, xi: torch.Tensor):
        """
        Initializes a disordered central Hamiltonian.

        Parameters:
        -----------
        Nx, Ny, t_x, t_y : same as Central.
        deltaV : torch.Tensor
            Amplitude range for the disorder potential.
        N_imp : int
            Number of impurities in the lattice.
        xi : torch.Tensor
            Correlation length of the disorder potential.
        """
        super().__init__(Nx, Ny, t_x, t_y)
        self.deltaV = deltaV
        self.N_imp = N_imp
        self.xi = xi

        # Add disorder potential
        self.H_full += self._construct_disorder_potential()

    def _construct_disorder_potential(self) -> torch.Tensor:
        """Generates the disorder potential for the lattice."""
        U = torch.zeros((self.Nx, self.Ny), dtype=torch.float32,device=self.funcDevice)

        # Generate random positions for scatterers
        R_k_x = torch.randint(0, self.Nx, (self.N_imp,), dtype=torch.int32,device=self.funcDevice)
        R_k_y = torch.randint(0, self.Ny, (self.N_imp,), dtype=torch.int32,device=self.funcDevice)

        # Generate random amplitudes for scatterers
        U_k = 2 * self.deltaV * torch.rand(self.N_imp) - self.deltaV

        # Compute the disorder potential for each lattice site
        for n in range(self.Nx):
            for j in range(self.Ny):
                r_nj = torch.tensor([n, j], dtype=torch.float32,device=self.funcDevice)
                for k in range(self.N_imp):
                    R_k = torch.tensor([R_k_x[k], R_k_y[k]], dtype=torch.float32,device=self.funcDevice)
                    U[n, j] += U_k[k] * torch.exp(-torch.norm(r_nj - R_k) ** 2 / (2 * self.xi ** 2))

        return torch.diag(U.flatten())


class CentralBdG(Central):
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, Delta: torch.Tensor):
        """
        Initializes a BdG central Hamiltonian with superconducting pairing.

        Parameters:
        ----------- 
        Ny : int
            Number of lattice sites in the y-direction.
        Nx : int
            Number of lattice sites in the x-direction.
        t_y : torch.Tensor
            Hopping parameter in the y-direction.
        t_x : torch.Tensor
            Hopping parameter in the x-direction.
        Delta : torch.Tensor
            Pairing potential for superconductivity.
        """
        super().__init__(Ny, Nx, t_y, t_x)
        self.Delta = Delta

        # Construct BdG Hamiltonian
        self.H_full_BdG = self._construct_bdg_with_pairing()

    def _construct_bdg_with_pairing(self) -> torch.Tensor:
        """Constructs the BdG Hamiltonian with superconducting pairing."""
        pairing_matrix = torch.eye(self.Ny * self.Nx, dtype=torch.complex64,device=self.funcDevice) * self.Delta
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64,device=self.funcDevice)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64,device=self.funcDevice)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64,device=self.funcDevice))+ \
                     torch.kron(pairing_matrix.conj(), torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64,device=self.funcDevice))
        return H_full_BdG

    def __repr__(self):
        return f"CentralBdG(Ny={self.Ny}, Nx={self.Nx}, t_y={self.t_y}, t_x={self.t_x}, Delta={self.Delta})"

class DisorderedCentralBdG(DisorderedCentral):
    def __init__(self, Ny, Nx, t_y, t_x, Delta, disorder_strength):
        super().__init__(Ny, Nx, t_y, t_x, disorder_strength)
        self.Delta = Delta
        self.H_full_BdG = self._construct_bdg_with_pairing()

    def _construct_bdg_with_pairing(self) -> torch.Tensor:
        """Constructs the BdG Hamiltonian with superconducting pairing."""
        pairing_matrix = torch.eye(self.Ny * self.Nx, dtype=torch.complex64, device=self.H_full.device) * self.Delta
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(pairing_matrix.conj(), torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64, device=self.H_full.device))
        return H_full_BdG


class CentralUniformDisorder:
    def __init__(self, Ny, Nx, t_y, t_x, U0, salt):
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.U0 = U0
        self.salt = salt
        self.funcDevice = t_x.device
        # Assemble the full Hamiltonian
        self.H_full = self._assemble_full_hamiltonian()

    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along the y-direction with disorder."""
        H_inter_y = self.t_y * torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64, device=self.funcDevice), 1)
        H_chain_y = H_inter_y + H_inter_y.T.conj()

        # Add disorder to the on-site terms using a vectorized approach
        disorder = self.U0 * (torch.rand(self.Ny, dtype=torch.complex64, device=self.funcDevice) - 0.5)
        H_chain_y += torch.diag(disorder)

        return H_chain_y

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        # Implementation of the full Hamiltonian assembly
        pass

class CentralBdGDisorder(CentralUniformDisorder):
    """# Example usage
    if __name__ == "__main__":
        Ny = 2
        Nx = 2
        t_y = torch.tensor(2.0, dtype=torch.complex64)
        t_x = torch.tensor(1.0, dtype=torch.complex64)
        Delta = torch.tensor(0, dtype=torch.complex64)
        U0 = 1.0
        salt = 13

        central_bdg_disorder = CentralBdGDisorder(Ny, Nx, t_y, t_x, Delta, U0, salt)
        H_full = central_bdg_disorder.H_full
        print(H_full)
    """
    def __init__(self, Ny, Nx, t_y, t_x, Delta, U0, salt):
        self.Delta = Delta
        super().__init__(Ny, Nx, t_y, t_x, U0, salt)

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        # Implementation of the full Hamiltonian assembly for BdG
        pass

    def H_full_BdG(self) -> torch.Tensor:
        # Implementation of the BdG Hamiltonian
        pass




class TopologicalSurface2D:
    """Class for constructing 2D topological surface state Hamiltonian."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor,
                 mu: torch.Tensor, B: torch.Tensor, M: torch.Tensor, 
                 boundary_condition: str = 'open'):
        """Initialize 2D topological surface state Hamiltonian.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Spin-orbit coupling parameters
            mu: Chemical potential
            M: Mass parameter
            boundary_condition: 'open', 'periodic', 'xperiodic', or 'yperiodic'
        """
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.mu = mu
        self.B = B
        self.M = M
        self.boundary_condition = boundary_condition.lower()
        self.funcDevice = t_x.device
        
        if self.boundary_condition not in ['open', 'periodic', 'xperiodic', 'yperiodic']:
            raise ValueError("boundary_condition must be 'open', 'periodic', 'xperiodic', or 'yperiodic'")
        
        # Pauli matrices in spin space
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct the full Hamiltonian
        self.H_full = self._construct_topological_hamiltonian()
        
    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along y-direction."""
        # Chemical potential and onsite terms
        H_onsite = self.B*(self.M)* self.sigma_z - self.mu * self.sigma_0
        H_chain = torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), 
                           H_onsite)
        
        # Nearest neighbor hopping along y
        t_nn_y = (4/3)*(self.sigma_z + 1j*self.sigma_y)/2
        H_nn_y = torch.kron(
            torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64, device=self.funcDevice), -1),
            t_nn_y
        )
        
        # Next-nearest neighbor hopping along y
        t_nnn_y = (-1/6) * (2 * self.sigma_z + 1j * self.sigma_y)/2
        H_nnn_y = torch.kron(
            torch.diag(torch.ones(self.Ny - 2, dtype=torch.complex64, device=self.funcDevice), -2),
            t_nnn_y
        )
        
        # Add periodic boundary conditions if needed
        if self.boundary_condition in ['periodic', 'yperiodic']:
            # Add hopping from last to first site (nearest neighbor)
            site1 = (self.Ny - 1) * 2
            site2 = 0
            H_nn_y[site2:site2+2, site1:site1+2] = t_nn_y
            
            # Add hopping from second-to-last to first site and last to second site (next-nearest)
            site1 = (self.Ny - 2) * 2
            H_nnn_y[site2:site2+2, site1:site1+2] = t_nnn_y
            site1 = (self.Ny - 1) * 2
            site2 = 2
            H_nnn_y[site2:site2+2, site1:site1+2] = t_nnn_y
        
        # Add all terms and their Hermitian conjugates
        H_chain = H_chain + H_nn_y + H_nn_y.conj().T + H_nnn_y + H_nnn_y.conj().T
        
        return H_chain
        
    def _construct_topological_hamiltonian(self) -> torch.Tensor:
        """Constructs the full 2D topological surface state Hamiltonian."""
        # Create hopping blocks first
        t_nn_x = (4/3)*(self.sigma_z + 1j * self.sigma_x)/2
        t_nnn_x = (-1/6) * (2 * self.sigma_z + 1j * self.sigma_x)/2
        
        # Create intermediate blocks
        block_t_nn_x = torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), t_nn_x)
        block_t_nnn_x = torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), t_nnn_x)
        
        # Construct hopping along x-direction
        H_nn_x = torch.kron(
            torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64, device=self.funcDevice), -1),
            block_t_nn_x
        )
        
        if self.Nx == 1:
            import warnings
            warnings.warn("Nx=1: Next-nearest neighbor hopping in x-direction is set to zero as there are no next-nearest neighbors")
            H_nnn_x = torch.zeros(2*self.Ny, 2*self.Ny, dtype=torch.complex64, device=self.funcDevice)
        else:
            H_nnn_x = torch.kron(
                torch.diag(torch.ones(self.Nx - 2, dtype=torch.complex64, device=self.funcDevice), -2),
                block_t_nnn_x
            )
        
        # Add periodic boundary conditions in x direction if needed
        if self.boundary_condition in ['periodic', 'xperiodic']:
            # Add hopping from last to first column (nearest neighbor)
            start_idx = 0
            end_idx = 2 * self.Ny
            last_idx = 2 * self.Ny * (self.Nx - 1)
            
            # Last to first (nearest neighbor)
            H_nn_x[start_idx:end_idx, last_idx:last_idx+end_idx] = block_t_nn_x
            
            # Second-to-last to first (next-nearest)
            second_last_idx = 2 * self.Ny * (self.Nx - 2)
            H_nnn_x[start_idx:end_idx, second_last_idx:second_last_idx+end_idx] = block_t_nnn_x
            
            # Last to second (next-nearest)
            second_idx = 2 * self.Ny
            H_nnn_x[second_idx:second_idx+end_idx, last_idx:last_idx+end_idx] = block_t_nnn_x
        
        # Full Hamiltonian with all terms
        H_full = torch.kron(torch.eye(self.Nx, dtype=torch.complex64, device=self.funcDevice), 
                           self._construct_chain_y()) + \
                 H_nn_x + H_nn_x.conj().T + H_nnn_x + H_nnn_x.conj().T
        
        return H_full


class MZMVortexHamiltonian(TopologicalSurface2D):
    """Class for constructing BdG Hamiltonian with Majorana zero mode vortices."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, 
                 Delta_0: torch.Tensor, xi_0: torch.Tensor, lambda_L: torch.Tensor, vortex_positions: List[Tuple[torch.Tensor, torch.Tensor]],
                 mu: torch.Tensor, B: torch.Tensor, M: torch.Tensor, boundary_condition: str = 'open'):
        """Initialize vortex Hamiltonian for 2D topological surface state.
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Spin-orbit coupling parameters
            mu: Chemical potential
            B: Magnetic field
            M: Mass parameter
            boundary_condition: 'open', 'periodic', 'xperiodic', or 'yperiodic'
        """
        super().__init__(Ny, Nx, t_y, t_x, mu,B, M, boundary_condition)
        self.Delta = Delta_0
        self.xi_0 = xi_0
        self.lambda_L = lambda_L.item()
        self.vortex_positions = vortex_positions
        self.hbar = torch.tensor(1, dtype=torch.float32, device=self.funcDevice)
        self.e = torch.tensor(1, dtype=torch.float32, device=self.funcDevice)
        self.Phi_0 = torch.pi * self.hbar / self.e # Half Magnetic flux quantum in natural units
        
        # Apply Peierls substitution to normal state Hamiltonian
        self.H_full = self._apply_peierls_substitution(self.H_full)
        
        # Construct full BdG Hamiltonian with vortices
        self.H_full_BdG = self._construct_vortex_hamiltonian()
        
    def _apply_peierls_substitution(self, h_normal: torch.Tensor) -> torch.Tensor:
        """Apply Peierls substitution to modify hopping terms with vector potential.
        """
        h_modified = h_normal.clone()
        
        # Loop over all sites
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                site1 = ix * self.Ny + iy
                x1, y1 = torch.tensor(ix, dtype=torch.float32, device=self.funcDevice), torch.tensor(iy, dtype=torch.float32, device=self.funcDevice)  # Actual lattice positions
                
                # Only process right and up hoppings
                neighbors = [
                    (ix+1, iy),   # Right only
                    (ix, iy+1),   # Up only
                ]
                
                # Only process right2 and up2 for next nearest neighbors
                nnn_neighbors = [
                    (ix+2, iy),   # Right 2 only
                    (ix, iy+2),   # Up 2 only
                ]
                
                # Process neighbors
                for nx, ny in neighbors + nnn_neighbors: 
                    x2, y2 = torch.tensor(nx, dtype=torch.float32, device=self.funcDevice), torch.tensor(ny, dtype=torch.float32, device=self.funcDevice)
       
                    # Calculate Peierls phase
                    phase = self._calculate_peierls_phase(x1, y1, x2, y2)

                                        # Handle periodic boundary conditions
                    if self.boundary_condition in ['periodic', 'xperiodic']:
                        if nx >= self.Nx:
                            nx = nx % self.Nx  # Wrap around x
                        if ny >= self.Ny:
                            continue
                    if self.boundary_condition in ['periodic', 'yperiodic']:
                        if ny >= self.Ny:
                            ny = ny % self.Ny  # Wrap around y
                        if nx >= self.Nx:
                            continue
                    if self.boundary_condition in ['open']:
                        if nx >= self.Nx or ny >= self.Ny:
                            continue
                    site2 = nx * self.Ny + ny
                              
                    
                    # Modify hopping terms for all spin combinations
                    for spin1 in [0, 1]:
                        for spin2 in [0, 1]:
                            idx1 = site1 * 2 + spin1
                            idx2 = site2 * 2 + spin2
                            if h_modified[idx2, idx1] != 0:
                                h_modified[idx2, idx1] *= phase
                                h_modified[idx1, idx2] = h_modified[idx2, idx1].conj()
        
        return h_modified
    
    def _calculate_delta(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate superconducting gap at position (x,y)."""
        delta = torch.ones(1, dtype=torch.complex64, device=self.funcDevice) * self.Delta
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                return torch.zeros(1, dtype=torch.complex64, device=self.funcDevice)[0]
            
            # Amplitude factor with tanh profile
            tanh_factor = torch.tanh(r / self.xi_0)
            
            # Phase factor (x+iy)/r for p-wave like vortex
            phase = ((x - x_j) + 1j*(y - y_j))/r
            
            delta *= tanh_factor * phase
            # delta *= phase
            # Convert delta to scalar tensor by taking first element
        return delta[0]
        # return torch.tensor(0.2, dtype=torch.complex64, device=self.funcDevice)

    def _calculate_vector_potential(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate vector potential at position (x,y)."""
        Ax = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        Ay = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                continue
                
            # Calculate K1 Bessel function
            from scipy.special import k1
            K1_factor = torch.tensor(k1(r.cpu().numpy() / self.lambda_L), 
                                   device=self.funcDevice)
            
            # Vector potential from London equation solution
            common = self.Phi_0/(2*torch.pi*r) * (1 - r/self.lambda_L * K1_factor)
            
            # Add contribution in polar coordinates
            Ax += -common * (y - y_j)/r  # -sin(θ) component
            Ay += common * (x - x_j)/r   # cos(θ) component
            
        return Ax, Ay
        
    def _calculate_peierls_phase(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Calculate Peierls phase between two points."""
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        Ax, Ay = self._calculate_vector_potential(x_mid, y_mid)
        dx = x2 - x1
        dy = y2 - y1
        phase = self.e * (Ax * dx + Ay * dy) / self.hbar
        # Convert phase to scalar tensor by taking first element
        return torch.exp(1j * phase)[0]
        # return torch.tensor(1, dtype=torch.complex64, device=self.funcDevice)
    def _construct_vortex_hamiltonian(self) -> torch.Tensor:
        """Construct full BdG Hamiltonian with vortices."""
        h_bdg = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64,device=self.funcDevice)) + \
                torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64,device=self.funcDevice)) 
        
        # Add pairing terms
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                pos = (ix * self.Ny + iy) * 2 * 2  # *2 for spin and 2 for BdG
                x, y = torch.tensor(ix, dtype=torch.float32, device=self.funcDevice), torch.tensor(iy, dtype=torch.float32, device=self.funcDevice)  # Use actual lattice positions
                delta = self._calculate_delta(x, y)
                # Pairing in the basis |c_↑†,c_↑,c_↓†,c_↓⟩
                h_bdg[pos, pos+3] = delta  # c_↑† to c_↓†
                h_bdg[pos+1, pos+2] = -delta.conj()  # c_↑ to c_↓
                h_bdg[pos+2, pos+1] = -delta  # c_↓† to c_↑†
                h_bdg[pos+3, pos] = delta.conj()  # c_↓ to c_↑
        
        return h_bdg

    @staticmethod
    def generate_triangular_vortex_lattice(L: int, n: int, m: int, device='cpu') -> List[Tuple[float, float]]:
        """Generate vortex positions in a triangular lattice pattern.
        
        This method creates vortex positions based on the description in the paper.
        The positions are centers of square lattice plaquettes (x+0.5, y+0.5).
        
        Args:
            L: Size of the square lattice (L×L)
            n, m: Parameters defining the triangular lattice vectors δ₁=(n,m) and δ₂=(m,n)
            device: Device for torch tensors
            
        Returns:
            List of vortex positions as (x, y) coordinates
        """
        # Verify that L is compatible with the (n, m) pair
        # For the (n,m) pairs mentioned in the paper, valid L values can be calculated
        # using the relationship (L,0) = k₁δ₁ + k₂δ₂
        
        # Find the values of k₁ and k₂
        # This means solving: L = k₁n + k₂m and 0 = k₁m + k₂n
        # From the second equation: k₂ = -k₁m/n
        # Substituting into the first: L = k₁n - k₁m²/n = k₁(n - m²/n)
        # Therefore: k₁ = L·n/(n² - m²)
        
        # Check if L is compatible
        if (n**2 - m**2) <= 0:
            raise ValueError(f"Invalid (n,m) pair: ({n},{m}). Must satisfy n² > m².")
        
        k1 = L * n / (n**2 - m**2)
        k2 = -k1 * m / n
        
        # Check if k1 and k2 are integers (within numerical precision)
        if not (abs(k1 - round(k1)) < 1e-10 and abs(k2 - round(k2)) < 1e-10):
            raise ValueError(f"Lattice size L={L} is not compatible with (n,m)=({n},{m}). Try different values.")
        
        k1, k2 = int(round(k1)), int(round(k2))
        
        # Calculate number of vortices in the unit cell
        det = n**2 - m**2  # Determinant of the transformation matrix
        nv = abs(det)
        
        # Check if number of vortices is even as required
        if nv % 2 != 0:
            print(f"Warning: Number of vortices ({nv}) is not even. Consider different parameters.")
        
        # Generate vortex positions based on the lattice vectors δ₁=(n,m) and δ₂=(m,n)
        vortex_positions = []
        
        # Define the lattice vectors
        delta1 = (n, m)
        delta2 = (m, n)
        
        # Compute the area of the unit cell
        cell_area = n**2 - m**2
        
        # Generate all possible positions within the unit cell
        for i in range(L):
            for j in range(L):
                # Convert to fractional coordinates within the basis of delta1 and delta2
                # This involves solving: (i,j) = a*delta1 + b*delta2
                # We have: i = a*n + b*m and j = a*m + b*n
                
                # Calculate coefficients in the new basis
                denominator = n**2 - m**2
                a = (i*n - j*m) / denominator
                b = (j*n - i*m) / denominator
                
                # Only include points that are inside the unit cell (0 <= a,b < 1)
                if 0 <= a < 1 and 0 <= b < 1:
                    # Place vortex at plaquette center
                    vortex_positions.append((i + 0.5, j + 0.5))
        
        # Verify we have the correct number of vortices
        if len(vortex_positions) != nv:
            print(f"Warning: Generated {len(vortex_positions)} vortices, expected {nv}.")
        
        return vortex_positions
        
    def visualize_vector_potential(self):
        """Visualize vector potential field using quiver plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create grid points
        x = np.linspace(0, self.Nx-1, self.Nx)
        y = np.linspace(0, self.Ny-1, self.Ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate vector potential at each point
        Ax = np.zeros((self.Ny, self.Nx))
        Ay = np.zeros((self.Ny, self.Nx))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_tensor = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y_tensor = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                Ax_t, Ay_t = self._calculate_vector_potential(x_tensor, y_tensor)
                Ax[j,i] = Ax_t.cpu().numpy()
                Ay[j,i] = Ay_t.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.quiver(X, Y, Ax, Ay)
        
        # Plot vortex positions
        vortex_x = [pos[0] for pos in self.vortex_positions]
        vortex_y = [pos[1] for pos in self.vortex_positions]
        plt.plot(vortex_x, vortex_y, 'ro', label='Vortices')
        
        plt.title('Vector Potential Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.colorbar()
        plt.savefig('vector_potential.png')
        plt.close()

    def visualize_delta_field(self):
        """Visualize superconducting order parameter."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate Delta at each point
        Delta_mag = np.zeros((self.Ny, self.Nx))
        Delta_phase = np.zeros((self.Ny, self.Nx))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_tensor = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y_tensor = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                delta = self._calculate_delta(x_tensor, y_tensor)
                Delta_mag[j,i] = abs(delta.cpu().numpy())
                Delta_phase[j,i] = np.angle(delta.cpu().numpy())
        
        # Plot magnitude
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        im1 = plt.imshow(Delta_mag, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-1])
        plt.colorbar(im1, label='|Δ|')
        plt.title('Superconducting Gap Magnitude')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot vortex positions
        vortex_x = [pos[0].cpu().numpy() for pos in self.vortex_positions]
        vortex_y = [pos[1].cpu().numpy() for pos in self.vortex_positions]
        plt.plot(vortex_x, vortex_y, 'wo', markeredgecolor='black', label='Vortices')
        plt.legend()
        
        # Plot phase
        plt.subplot(122)
        im2 = plt.imshow(Delta_phase, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-1], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, label='arg(Δ)')
        plt.title('Superconducting Phase')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(vortex_x, vortex_y, 'wo', markeredgecolor='black', label='Vortices')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('delta_field.png')
        plt.close()

    def visualize_peierls_phase(self):
        """Visualize Peierls phase for nearest-neighbor hoppings."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate Peierls phase for horizontal and vertical bonds
        phase_x = np.zeros((self.Ny, self.Nx-1))  # horizontal bonds
        phase_y = np.zeros((self.Ny-1, self.Nx))  # vertical bonds
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x1 = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y1 = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                
                # Horizontal bonds
                if i < self.Nx-1:
                    x2 = torch.tensor(float(i+1), dtype=torch.float32, device=self.funcDevice)
                    phase = self._calculate_peierls_phase(x1, y1, x2, y1)
                    phase_x[j,i] = np.angle(phase.cpu().numpy())
                
                # Vertical bonds
                if j < self.Ny-1:
                    y2 = torch.tensor(float(j+1), dtype=torch.float32, device=self.funcDevice)
                    phase = self._calculate_peierls_phase(x1, y1, x1, y2)
                    phase_y[j,i] = np.angle(phase.cpu().numpy())
        
        # Plot phases
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        im1 = plt.imshow(phase_x, origin='lower', extent=[0, self.Nx-2, 0, self.Ny-1], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im1, label='Phase (rad)')
        plt.title('Peierls Phase - Horizontal Bonds')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(122)
        im2 = plt.imshow(phase_y, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-2], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, label='Phase (rad)')
        plt.title('Peierls Phase - Vertical Bonds')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        plt.savefig('peierls_phase.png')
        plt.close()

class ChernTexturedInsulator(TopologicalSurface2D):
    """
    Implements a Chern textured insulator Hamiltonian with two valleys of opposite Chern numbers
    and pairing between them.
    
    This creates a 4×4 matrix structure per site:
    - 2×2 from the original spin degree of freedom
    - Another factor of 2 from the valley (Chern number) degree of freedom
    
    The Hamiltonian has the form:
    H = H_positive ⊗ [[1,0],[0,0]] + H_negative ⊗ [[0,0],[0,1]] + Delta * sigma_z ⊗ tau_z
    """
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor,
                 mu: torch.Tensor, B: torch.Tensor, M: torch.Tensor, Delta: torch.Tensor,
                 boundary_condition: str = 'open', 
                 valley_coupling: torch.Tensor = None):
        """
        Initialize the Chern textured insulator Hamiltonian.
        
        Args:
            Ny: Number of sites in y-direction
            Nx: Number of sites in x-direction
            t_y: Hopping parameter in y-direction
            t_x: Hopping parameter in x-direction
            mu: Chemical potential
            B: Zeeman field strength
            M: Mass term determining the Chern number (M > 0 for positive, -M for negative)
            Delta: Pairing strength between valleys
            boundary_condition: 'open', 'periodic', 'xperiodic', or 'yperiodic'
            valley_coupling: Optional coupling between valleys (default: None)
        """
        # Initialize the parent class with the base parameters
        super().__init__(Ny, Nx, t_y, t_x, mu, B, M, boundary_condition)
        
        # Store additional parameters
        self.Delta = Delta
        self.valley_coupling = valley_coupling
        
        # Define Pauli matrices for the valley degree of freedom (tau)
        self.tau_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct the full Hamiltonian with both valleys and pairing
        self.H_full = self._construct_textured_hamiltonian()
    
    def _construct_positive_chern_hamiltonian(self) -> torch.Tensor:
        """Construct Hamiltonian for positive Chern number (M > 0)"""
        # We can use the parent class's method directly for positive Chern number
        return super()._construct_topological_hamiltonian()
    
    def _construct_negative_chern_hamiltonian(self) -> torch.Tensor:
        """Construct Hamiltonian for negative Chern number (M < 0)"""
        # Temporarily flip the sign of M and construct the Hamiltonian
        original_M = self.M
        self.M = -self.M  # Flip the sign to get opposite Chern number
        H_negative = super()._construct_topological_hamiltonian()
        self.M = original_M  # Restore original M
        return H_negative
    
    def _construct_textured_hamiltonian(self) -> torch.Tensor:
        """
        Construct the full Chern textured insulator Hamiltonian using tensor products.
        
        Returns:
            torch.Tensor: The full Hamiltonian combining both valleys and pairing.
        """
        # Get individual valley Hamiltonians
        H_positive = self._construct_positive_chern_hamiltonian()
        H_negative = self._construct_negative_chern_hamiltonian()
        
        # Valley projectors (tau space)
        valley_plus = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=self.funcDevice)
        valley_minus = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=self.funcDevice)
        
        # First part: Valley Hamiltonians
        H_textured = torch.kron(H_positive, valley_plus) + torch.kron(H_negative, valley_minus)
        
        # Second part: Pairing term Delta * sigma_z ⊗ tau_z
        # Create sigma_z for each site using kronecker products
        sigma_x_site = self.sigma_x  # Already defined in parent class
        
        # Expand sigma_z to the full lattice
        sigma_x_lattice = torch.eye(self.Nx * self.Ny, dtype=torch.complex64, device=self.funcDevice)
        sigma_x_full = torch.kron(sigma_x_lattice, sigma_x_site)
        
        # Add pairing term
        H_textured += self.Delta * torch.kron(sigma_x_full, self.tau_x)
        
        # Add valley coupling if provided
        if self.valley_coupling is not None:
            H_textured += self.valley_coupling * torch.kron(H_positive, self.tau_x)
        
        return H_textured
    
    @property
    def H(self) -> torch.Tensor:
        """Return the full Hamiltonian."""
        return self.H_full
    
    def __repr__(self):
        return (f"ChernTexturedInsulator(Nx={self.Nx}, Ny={self.Ny}, M={self.M}, "
                f"Delta={self.Delta}, boundary={self.boundary_condition})")
    
    @staticmethod
    def generate_triangular_vortex_config(config_name: str) -> Tuple[int, List[Tuple[float, float]]]:
        """Generate predefined triangular vortex lattice configurations from the paper.
        
        Args:
            config_name: Name of the configuration, must be one of:
                         'small': (n,m)=(6,2), L=16, nv=8
                         'medium': (n,m)=(11,3), L=112, nv=112
                         '3_1': (n,m)=(3,1), L=8, nv=8
                         '15_4': (n,m)=(15,4), L=209, nv=224
            
        Returns:
            Tuple containing (L, vortex_positions)
            L: Size of the square lattice (L×L)
            vortex_positions: List of vortex positions as (x, y) coordinates
        """
        # Define known valid configurations from the paper
        valid_configs = {
            'small': {'n': 6, 'm': 2, 'L': 16, 'nv': 32},  # (n,m)=(6,2), L=16, nv=32
            'medium': {'n': 11, 'm': 3, 'L': 112, 'nv': 112},  # (n,m)=(11,3), L=112, nv=112
            '3_1': {'n': 3, 'm': 1, 'L': 8, 'nv': 8},  # (n,m)=(3,1), L=8, nv=8
            '15_4': {'n': 15, 'm': 4, 'L': 209, 'nv': 209}  # (n,m)=(15,4), L=209, nv=209
        }
        
        if config_name not in valid_configs:
            raise ValueError(f"Unknown configuration: {config_name}. Must be one of: {list(valid_configs.keys())}")
        
        config = valid_configs[config_name]
        n, m, L = config['n'], config['m'], config['L']
        
        # Calculate vortex positions for this configuration
        vortex_positions = []
        
        # Determinant of the transformation matrix
        det = n**2 - m**2
        
        # Generate all possible positions within the unit cell
        for i in range(L):
            for j in range(L):
                # Calculate coefficients in the new basis
                denominator = n**2 - m**2
                a = (i*n - j*m) / denominator
                b = (j*n - i*m) / denominator
                
                # Only include points that are inside the unit cell (0 <= a,b < 1)
                if 0 <= a < 1 and 0 <= b < 1:
                    # Place vortex at plaquette center
                    vortex_positions.append((i + 0.5, j + 0.5))
        
        # Quick verification
        if len(vortex_positions) != config['nv']:
            print(f"Warning: Generated {len(vortex_positions)} vortices, expected {config['nv']}.")
            
        return L, vortex_positions

class SSHChain:
    """Base class for constructing SSH chain Hamiltonian."""
    
    def __init__(self, Nx_cell: int, t_u: torch.Tensor, t_v: torch.Tensor, 
                 mu: torch.Tensor, splitting_onsite_energy: torch.Tensor):
        """Initialize SSH chain Hamiltonian.
        
        Args:
            Nx_cell: Number of unit cells
            t_u: Inter-cell hopping strength
            t_v: Intra-cell hopping strength
            mu: Chemical potential
            splitting_onsite_energy: Onsite energy splitting at the ends
        """
        self.Nx_cell = Nx_cell
        self.t_u = t_u
        self.t_v = t_v
        self.mu = mu
        self.splitting_onsite_energy = splitting_onsite_energy
        self.funcDevice = t_u.device
        
        # Construct the full Hamiltonian
        self.H_full = self._construct_ssh_hamiltonian()
        
    def _construct_ssh_hamiltonian(self) -> torch.Tensor:
        """Constructs the SSH chain Hamiltonian."""
        # Total number of sites
        Nx = self.Nx_cell * 2
        
        # Initialize the Hamiltonian matrix
        H = torch.zeros((Nx, Nx), dtype=torch.complex64, device=self.funcDevice)
        
        # Add on-site terms (chemical potential)
        for i in range(Nx):
            H[i, i] = -self.mu
        
        # Add hopping terms
        for i in range(Nx-1):
            if i % 2 == 0:  # Intra-cell hopping (t_v)
                H[i, i+1] = -self.t_v
                H[i+1, i] = -self.t_v.conj()  # Hermitian conjugate
            else:  # Inter-cell hopping (t_u)
                H[i, i+1] = -self.t_u
                H[i+1, i] = -self.t_u.conj()  # Hermitian conjugate
        
        # Add splitting onsite energy at the ends
        H[0, 0] += self.splitting_onsite_energy
        H[Nx-1, Nx-1] -= self.splitting_onsite_energy
        
        return H
    
    def __repr__(self):
        return f"SSHChain(Nx_cell={self.Nx_cell}, t_u={self.t_u}, t_v={self.t_v}, mu={self.mu}, splitting_onsite_energy={self.splitting_onsite_energy})"


class SSHChainBdG(SSHChain):
    """Class for constructing SSH chain Hamiltonian with BdG formalism."""
    
    def __init__(self, Nx_cell: int, t_u: torch.Tensor, t_v: torch.Tensor, 
                 mu: torch.Tensor, Delta: torch.Tensor, splitting_onsite_energy: torch.Tensor):
        """Initialize SSH chain Hamiltonian with BdG formalism.
        
        Args:
            Nx_cell: Number of unit cells
            t_u: Inter-cell hopping strength
            t_v: Intra-cell hopping strength
            mu: Chemical potential
            Delta: Pairing potential
            splitting_onsite_energy: Onsite energy splitting at the ends
        """
        self.Delta = Delta
        super().__init__(Nx_cell, t_u, t_v, mu, splitting_onsite_energy)
        
        # Construct BdG Hamiltonian
        self.H_full_BdG = self._construct_bdg_with_pairing()
        
    def _construct_bdg_with_pairing(self) -> torch.Tensor:
        """Constructs the BdG Hamiltonian with superconducting pairing."""
        # Create pairing matrix
        Nx = self.Nx_cell * 2
        pairing_matrix = torch.diag(torch.ones(Nx-1, dtype=torch.complex64, device=self.funcDevice),1) * self.Delta
        
        # Construct BdG Hamiltonian following the same pattern as CentralBdG
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=self.funcDevice)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=self.funcDevice)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=self.funcDevice)) + \
                     torch.kron(pairing_matrix.conj(), torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64, device=self.funcDevice))
        
        return H_full_BdG
class SSH2DCellMethod:
    """
    Class for constructing the 2D SSH Hamiltonian based on the provided formula
    and diagram [Fig. 21(a)], featuring pi-flux per plaquette.
    Each unit cell R = (ix, iy) contains 4 sites labeled 1, 2, 3, 4.
    Note Nx,Ny is the number of unit cells, not the number of sites.
    """
    def __init__(self, Nx: int, Ny: int, 
                 gamma_x: torch.Tensor, gamma_y: torch.Tensor,
                 lambda_x: torch.Tensor, lambda_y: torch.Tensor):
        """
        Initialize the 2D SSH Hamiltonian.

        Args:
            Nx (int): Number of unit cells in the x-direction.
            Ny (int): Number of unit cells in the y-direction.
            gamma_x (torch.Tensor): Hopping amplitude within cell along x (1<->3, 2<->4).
            gamma_y (torch.Tensor): Hopping amplitude within cell along y (1<->4, 2<->3, with sign).
            lambda_x (torch.Tensor): Hopping amplitude between cells along x (1->3(R+x), 4->2(R+x)).
            lambda_y (torch.Tensor): Hopping amplitude between cells along y (1->4(R+y), 3->2(R+y), with sign).
        """
        self.Nx = Nx
        self.Ny = Ny
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.funcDevice = gamma_x.device # Assume all tensors are on the same device

        # Total number of sites = Nx * Ny * 4
        self.N_sites = self.Nx * self.Ny * 4

        # Construct the Hamiltonian
        self.H_full = self._construct_hamiltonian()

    def _get_site_index(self, ix: int, iy: int, site_in_cell: int) -> int:
        """
        Calculates the flattened index for a site.
        Args:
            ix (int): x-index of the unit cell (0 to Nx-1).
            iy (int): y-index of the unit cell (0 to Ny-1).
            site_in_cell (int): Site index within the unit cell (1, 2, 3, or 4).
        Returns:
            int: The global index in the flattened Hamiltonian matrix (0 to N_sites-1).
        """
        if not (0 <= ix < self.Nx and 0 <= iy < self.Ny):
            raise ValueError(f"Unit cell index ({ix}, {iy}) out of bounds ({self.Nx}x{self.Ny})")
        if not (1 <= site_in_cell <= 4):
            raise ValueError(f"Site index within cell must be 1, 2, 3, or 4, got {site_in_cell}")

        # Linear index for the unit cell
        cell_index = ix * self.Ny + iy
        # Global site index
        return cell_index * 4 + (site_in_cell - 1)

    def _construct_hamiltonian(self) -> torch.Tensor:
        """Constructs the full 2D SSH Hamiltonian matrix."""
        H = torch.zeros((self.N_sites, self.N_sites), dtype=torch.complex64, device=self.funcDevice)

        for ix in range(self.Nx):
            for iy in range(self.Ny):
                # --- Intra-cell hopping (gamma_x, gamma_y) ---
                idx1 = self._get_site_index(ix, iy, 1)
                idx2 = self._get_site_index(ix, iy, 2)
                idx3 = self._get_site_index(ix, iy, 3)
                idx4 = self._get_site_index(ix, iy, 4)

                # gamma_x terms: (c1† c3 + c2† c4 + H.c.)
                H[idx3, idx1] += self.gamma_x  # c1† c3 term corresponds to H[row=final, col=initial]
                H[idx1, idx3] += self.gamma_x.conj()
                H[idx2, idx4] += self.gamma_x
                H[idx4, idx2] += self.gamma_x.conj()

                # gamma_y terms: (c1† c4 - c2† c3 + H.c.)
                H[idx1, idx4] += self.gamma_y
                H[idx4, idx1] += self.gamma_y.conj()
                H[idx3, idx2] += -self.gamma_y  # Note the minus sign
                H[idx2, idx3] += (-self.gamma_y).conj()

                # --- Inter-cell hopping (lambda_x, lambda_y) ---

                # lambda_x terms: Connect R to R+x = (ix+1, iy)
                if ix < self.Nx - 1:
                    idx1_Rx = self._get_site_index(ix + 1, iy, 1)
                    idx2_Rx = self._get_site_index(ix + 1, iy, 2)
                    idx3_Rx = self._get_site_index(ix + 1, iy, 3)
                    idx4_Rx = self._get_site_index(ix + 1, iy, 4)

                    # (cR,1† cR+x,3 + cR,4† cR+x,2 + H.c.)
                    H[idx1, idx3_Rx] += self.lambda_x
                    H[idx3_Rx, idx1] += self.lambda_x.conj()
                    H[idx4, idx2_Rx] += self.lambda_x
                    H[idx2_Rx, idx4] += self.lambda_x.conj()


                # lambda_y terms: Connect R to R+y = (ix, iy+1)
                if iy < self.Ny - 1:
                    idx1_Ry = self._get_site_index(ix, iy + 1, 1)
                    idx2_Ry = self._get_site_index(ix, iy + 1, 2)
                    idx3_Ry = self._get_site_index(ix, iy + 1, 3)
                    idx4_Ry = self._get_site_index(ix, iy + 1, 4)

                    # (cR,1† cR+y,4 - cR,3† cR+y,2 + H.c.)
                    H[idx4, idx1_Ry] += self.lambda_y # cR,1† cR+y,4
                    H[idx1_Ry, idx4] += self.lambda_y.conj()
                    H[idx2, idx3_Ry] += -self.lambda_y # -cR,3† cR+y,2 (Note the minus sign)
                    H[idx3_Ry, idx2] += (-self.lambda_y).conj()

        return H

    def __repr__(self):
        return (f"SSH2D(Nx={self.Nx}, Ny={self.Ny}, gamma_x={self.gamma_x}, gamma_y={self.gamma_y}, "
                f"lambda_x={self.lambda_x}, lambda_y={self.lambda_y})")
class SSH2DChainMethod:
    """
    use slice or chain method to construct the 2D SSH Hamiltonian.
    """
    def __init__(self, Nx_cell: int, Ny_cell: int, 
                 gamma_x: torch.Tensor, gamma_y: torch.Tensor,
                 lambda_x: torch.Tensor, lambda_y: torch.Tensor):
        """
        Initialize the 2D SSH Hamiltonian.

        Args:
            Nx_cell (int): Number of unit cells in the x-direction.
            Ny_cell (int): Number of unit cells in the y-direction.
            gamma_x (torch.Tensor): Hopping amplitude within cell along x (1<->3, 2<->4).
            gamma_y (torch.Tensor): Hopping amplitude within cell along y (1<->4, 2<->3, with sign).
            lambda_x (torch.Tensor): Hopping amplitude between cells along x (1->3(R+x), 4->2(R+x)).
            lambda_y (torch.Tensor): Hopping amplitude between cells along y (1->4(R+y), 3->2(R+y), with sign).
        """
        self.Nx_cell = Nx_cell
        self.Ny_cell = Ny_cell
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.funcDevice = gamma_x.device # Assume all tensors are on the same device

        # Construct the Hamiltonian
        self.H_full = self._construct_hamiltonian()
    def _construct_chain_along_y(self,mu: torch.Tensor, t_u: torch.Tensor, t_v: torch.Tensor, Ny_cell: int) -> torch.Tensor:
        """
        Construct the Hamiltonian along the y-direction using the SSH chain method.
        
        Args:
            t_u: Inter-cell hopping strength
            t_v: Intra-cell hopping strength
            Ny_cell: Number of unit cells in y-direction
        
        Returns:
            torch.Tensor: The Hamiltonian matrix for the SSH chain along y-direction.
        """
        # Total number of sites
        Ny = Ny_cell * 2     
        # Initialize the Hamiltonian matrix
        H = torch.zeros((Ny, Ny), dtype=torch.complex64, device=self.funcDevice)

        # Add on-site terms (chemical potential)
        for i in range(Ny):
            H[i, i] = -mu
        
        # Add hopping terms
        for i in range(Ny-1):
            if i % 2 == 0:  # Intra-cell hopping (t_v)
                H[i, i+1] = -t_v
                H[i+1, i] = -t_v.conj()  # Hermitian conjugate
            else:  # Inter-cell hopping (t_u)
                H[i, i+1] = -t_u
                H[i+1, i] = -t_u.conj()  # Hermitian conjugate      
        return H
    def _construct_hamiltonian(self):
        """
        Construct the total Hamiltonian by combining the chain along y-direction with the chain along x-direction.
        """
        H_y_minus = self._construct_chain_along_y(torch.tensor(0.0, device=self.funcDevice), self.lambda_y, self.gamma_y, self.Ny_cell)
        H_y_plus = self._construct_chain_along_y(torch.tensor(0.0, device=self.funcDevice), -self.lambda_y, -self.gamma_y, self.Ny_cell)
        H_x_connect = self._construct_chain_along_y(torch.tensor(0.0, device=self.funcDevice), -self.lambda_x, -self.gamma_x, self.Nx_cell)
        H_total_diag = torch.kron(torch.eye(self.Nx_cell, device=self.funcDevice), torch.block_diag(H_y_minus, H_y_plus)) 
        H_total_offdiag = torch.kron(H_x_connect, torch.eye(2*self.Ny_cell, device=self.funcDevice))
        H_total = H_total_diag + H_total_offdiag
        return H_total
    def __repr__(self):
        return (f"SSH2DChain(Nx_cell={self.Nx_cell}, Ny_cell={self.Ny_cell}, "
                f"gamma_x={self.gamma_x}, gamma_y={self.gamma_y}, "
                f"lambda_x={self.lambda_x}, lambda_y={self.lambda_y})")
