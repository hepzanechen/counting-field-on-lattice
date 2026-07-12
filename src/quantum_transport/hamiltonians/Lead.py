import torch
from typing import Optional, Tuple, List, Union

class Lead:
    def __init__(self, mu: torch.Tensor, 
                 t_lead: torch.Tensor, 
                 connection_coordinates: List[Tuple[int, int]],
                 central_Nx: int,
                 central_Ny: int,
                 device: torch.device = None,
                 temperature: torch.Tensor = None,
                 t_lead_central: torch.Tensor = None):
        """
        Base Lead class inherited for different types of leads.
        
        Parameters:
        -----------
        mu : torch.Tensor
            Chemical potential for the lead.
        t_lead : torch.Tensor
            Hopping parameter within the lead.
        connection_coordinates : List[Tuple[int, int]]
            List of (x,y) coordinates where lead connects to the central region.
        central_Nx : int
            Number of sites in x-direction of the central sample.
        central_Ny : int
            Number of sites in y-direction of the central sample.
        device : torch.device, optional
            Device to place tensors on.
        temperature : torch.Tensor, optional
            Temperature.
        t_lead_central : torch.Tensor, optional
            Coupling strength between lead and central region.
        """
        # Set the device
        self.funcDevice = device if device is not None else mu.device
        
        # Ensure tensors are on the correct device
        self.mu = mu.to(self.funcDevice)
        self.temperature = temperature.to(self.funcDevice) if temperature is not None else torch.tensor(0.0, dtype=torch.float32, device=self.funcDevice)
        self.lambda_ = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.funcDevice)
        
        # Store central sample dimensions for reference
        self.central_Nx = central_Nx
        self.central_Ny = central_Ny
        
        # Initialize intraCell based on the lead type
        self.initialize_intracell()
        
        # Get the orbital multiplier for this lead type
        orbital_multiplier = self.get_orbital_multiplier()
        
        # Convert coordinates to site indices in the central sample
        site_indices = []
        for x, y in connection_coordinates:
            site_index = int(x) * central_Ny + int(y)
            site_indices.extend([site_index * orbital_multiplier + i for i in range(orbital_multiplier)])
        
        self.position = torch.tensor(site_indices, device=self.funcDevice)
        
        # Store for reference
        self.connection_coordinates = connection_coordinates
        
        # Infer lead width from the number of unique connection points
        unique_connection_points = set(connection_coordinates)
        self.lead_width = len(unique_connection_points)
        
        # Set up the lead matrices
        
        # leads inter-slice hopping
        self.t = t_lead.to(self.funcDevice) * torch.kron(
            torch.eye(self.lead_width, dtype=torch.complex64, device=self.funcDevice),
            self.intralCell
        )
        
        # leads intra-slice epsilon0
        if self.lead_width > 1:
            lead_inter_chains = t_lead.to(self.funcDevice) * torch.kron(
                torch.diag(torch.ones(self.lead_width - 1, device=self.funcDevice), 1),
                self.intralCell
            )
            self.epsilon0 = lead_inter_chains + lead_inter_chains.T.conj()
        else:
            # For single-width leads, no inter-chain connections
            self.epsilon0 = torch.zeros_like(self.t)
        
        
        # leads-central coupling
        self.t_lead_central = t_lead_central.to(self.funcDevice) if t_lead_central is not None else t_lead.to(self.funcDevice)
        
        # Create V1alpha coupling matrix
        self.V1alpha = self.t_lead_central * torch.kron(
            torch.eye(self.lead_width, dtype=torch.complex64, device=self.funcDevice),
            self.intralCell
        )
    
    def initialize_intracell(self):
        """Initialize lead structure - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_intracell")
    
    def get_orbital_multiplier(self) -> int:
        """Return the number of states per site - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_orbital_multiplier")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mu={self.mu}, temperature={self.temperature}, lead_width={self.lead_width})"


class SpinlessLead(Lead):
    """Lead class for spinless systems (single orbital per site)."""
    
    def initialize_intracell(self):
        """Initialize single-orbital structure."""
        self.intralCell = torch.tensor(1, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for spinless case."""
        return 1


class SpinfulLead(Lead):
    """Lead class for systems with spin (two spin states per site)."""
    
    def initialize_intracell(self):
        """Initialize spin-1/2 structure."""
        self.intralCell = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for spinful case."""
        return 2


class MultiOrbitalLead(Lead):
    """Lead class for systems with multiple orbitals per site."""
    
    def __init__(self, mu: torch.Tensor, 
                 t_lead: torch.Tensor, 
                 connection_coordinates: List[Tuple[int, int]],
                 central_Nx: int,
                 central_Ny: int,
                 num_orbitals: int,
                 device: torch.device = None,
                 temperature: torch.Tensor = None,
                 t_lead_central: torch.Tensor = None):
        """
        Initialize MultiOrbitalLead with specified number of orbitals.
        
        Parameters:
        -----------
        num_orbitals : int
            Number of orbitals per site.
        
        Other parameters are the same as the Lead class.
        """
        self.num_orbitals = num_orbitals
        super().__init__(mu, t_lead, connection_coordinates, central_Nx, central_Ny, 
                         device, temperature, t_lead_central)
    
    def initialize_intracell(self):
        """Initialize multi-orbital structure."""
        self.intralCell = torch.eye(self.num_orbitals, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for multi-orbital case."""
        return self.num_orbitals
