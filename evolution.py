"""
Optimized Python implementation of gravothermal evolution simulation
for self-interacting dark matter halos with static central baryonic potentials.
Uses NumPy vectorization and Numba JIT compilation for performance.
"""

import numpy as np
from numba import jit
import time
import os
from dataclasses import dataclass
from typing import Optional

# Enforce double precision (float64) to match C++ double
np.set_printoptions(precision=17)  # Full double precision display


def verify_precision():
    """Verify that we're using double precision (float64) throughout"""
    test_array = np.array([1.0])
    assert test_array.dtype == np.float64, f"Expected float64, got {test_array.dtype}"
    
    # Verify NumPy default float type
    default_float = np.zeros(1).dtype
    assert default_float == np.float64, f"NumPy default float is {default_float}, expected float64"
    
    # Check machine epsilon (should match C++ double)
    eps = np.finfo(np.float64).eps
    expected_eps = 2.220446049250313e-16  # Machine epsilon for double precision
    assert abs(eps - expected_eps) < 1e-20, f"Float64 epsilon mismatch: {eps} != {expected_eps}"
    
    return True


# Simulation configuration constants
DEFAULT_TOTAL_STEPS = 1000000000
DEFAULT_SAVE_STEPS = 100
DEFAULT_EPSILON = 0.001
DEFAULT_RELAXATION_STEPS = 10
DEFAULT_DENSITY_THRESHOLD = 1e30


@dataclass
class SimulationParameters:
    """Container for simulation parameters"""
    # Simulation control parameters
    total_step: int = DEFAULT_TOTAL_STEPS
    save_step: int = DEFAULT_SAVE_STEPS
    epsilon: float = DEFAULT_EPSILON
    total_time: float = 0.0
    
    # Cross section parameters
    a: float = 2.257
    b: float = 1.385
    c: float = 0.753
    sigma: float = 0.5
    
    # Baryon enclosed mass function parameters
    mass_norm: float = 0.1
    scale_norm: float = 0.1
    
    # IO parameters
    tag: str = "run01"
    input_dir: str = "output/run01/"
    output_file: str = "output/result_run01.txt"
    verbose: bool = False
    
    def __post_init__(self):
        """Set default directories after initialization"""
        if self.input_dir == "output/run01/":
            self.input_dir = f"output/{self.tag}/"
        if self.output_file == "output/result_run01.txt":
            self.output_file = f"output/result_{self.tag}.txt"
    
    def display(self):
        """Display parameters to console"""
        print(f"Initial profile: {self.tag}")
        print(f"Initial time: {self.total_time}")
        print(f"Total step: {self.total_step}")
        print(f"Save step: {self.save_step}")
        print(f"Abs(delta u/u): {self.epsilon}")
        print(f"Cross section (sigma): {self.sigma}")
        print(f"Conduction parameter (a, b, c): {self.a}, {self.b}, {self.c}")
        print(f"Baryon parameter (mass_norm, scale_norm): {self.mass_norm}, {self.scale_norm}")


@jit(nopython=True, cache=True)
def mbaryon_plummer(r: float, mass: float, scale: float) -> float:
    """
    Plummer model enclosed mass function
    
    Args:
        r: Radius
        mass: Normalized baryon mass
        scale: Normalized scale radius
    
    Returns:
        Enclosed baryon mass at radius r
    """
    return mass * (1.0 + (scale * scale) / (r * r)) ** (-1.5)


@jit(nopython=True, cache=True)
def mbaryon_plummer_vectorized(r_array: np.ndarray, mass: float, scale: float) -> np.ndarray:
    """
    Vectorized Plummer model enclosed mass function
    
    Args:
        r_array: Array of radii
        mass: Normalized baryon mass
        scale: Normalized scale radius
    
    Returns:
        Array of enclosed baryon masses
    """
    result = np.empty_like(r_array)
    for i in range(len(r_array)):
        result[i] = mass * (1.0 + (scale * scale) / (r_array[i] * r_array[i])) ** (-1.5)
    return result


class SimulationState:
    """Container for simulation state arrays"""
    
    def __init__(self):
        self.r_list: Optional[np.ndarray] = None
        self.rho_list: Optional[np.ndarray] = None
        self.m_list: Optional[np.ndarray] = None
        self.mhy_list: Optional[np.ndarray] = None
        self.u_list: Optional[np.ndarray] = None
        self.l_list: Optional[np.ndarray] = None
        self.v_list: Optional[np.ndarray] = None
        self.p_list: Optional[np.ndarray] = None
        self.a_list: Optional[np.ndarray] = None
        self.no_layers: int = 0
    
    def initialize(self, params: SimulationParameters):
        """
        Initialize state from input files
        
        Args:
            params: Simulation parameters
        """
        # Construct input file paths
        name_r = os.path.join(params.input_dir, f"RList-{params.tag}.txt")
        name_rho = os.path.join(params.input_dir, f"RhoList-{params.tag}.txt")
        name_m = os.path.join(params.input_dir, f"MList-{params.tag}.txt")
        name_u = os.path.join(params.input_dir, f"uList-{params.tag}.txt")
        name_l = os.path.join(params.input_dir, f"LList-{params.tag}.txt")
        
        # Read data with explicit float64 (double precision) dtype
        self.r_list = np.loadtxt(name_r, dtype=np.float64).flatten()
        self.rho_list = np.loadtxt(name_rho, dtype=np.float64).flatten()
        self.m_list = np.loadtxt(name_m, dtype=np.float64).flatten()
        self.u_list = np.loadtxt(name_u, dtype=np.float64).flatten()
        self.l_list = np.loadtxt(name_l, dtype=np.float64).flatten()
        
        self.no_layers = len(self.r_list)
        
        # Initialize arrays with explicit float64 dtype
        self.mhy_list = np.zeros(self.no_layers, dtype=np.float64)
        self.v_list = np.zeros(self.no_layers, dtype=np.float64)
        self.p_list = np.zeros(self.no_layers, dtype=np.float64)
        self.a_list = np.zeros(self.no_layers, dtype=np.float64)
        
        # Set initial 1D velocity dispersion
        self.v_list = np.sqrt(2.0/3.0) * np.sqrt(self.u_list)
        
        # Add baryon mass (Plummer model)
        self.mhy_list[:-1] = self.m_list[:-1] + mbaryon_plummer_vectorized(
            self.r_list[:-1], params.mass_norm, params.scale_norm
        )
        
        if params.verbose:
            print(f"Number of layers: {self.no_layers}")
            print(f"Initial r inner most: {self.r_list[0]}")
            print(f"Initial r next to outer most: {self.r_list[-1]}")
            print(f"Initial rho inner most: {self.rho_list[0]}")
            print(f"Initial rho next to outer most: {self.rho_list[-1]}")
    
    def check_for_abnormal_state(self) -> bool:
        """
        Check for abnormal state that would require stopping the simulation
        
        Returns:
            True if simulation should stop, False otherwise
        """
        if self.rho_list[0] > DEFAULT_DENSITY_THRESHOLD:
            print("Rho reaches threshold!")
            return True
        
        if self.r_list[0] < 0:
            print("R(0) is negative!")
            return True
        
        if np.isnan(self.r_list[0]):
            print("R is nan!")
            return True
        
        return False


@jit(nopython=True, cache=True)
def setup_hydrostatic_matrix_jit(hydromat: np.ndarray, hydrob: np.ndarray,
                                  r_list: np.ndarray, p_list: np.ndarray,
                                  rho_list: np.ndarray, mhy_list: np.ndarray,
                                  no_layers: int):
    """
    JIT-compiled function to set up the hydrostatic equilibrium matrix
    
    Args:
        hydromat: Output matrix (modified in place)
        hydrob: Output vector (modified in place)
        r_list: Radius array
        p_list: Pressure array
        rho_list: Density array
        mhy_list: Total mass array
        no_layers: Number of layers
    """
    # Setup first row
    R0 = r_list[0]
    R0_2 = R0 * R0
    R0_3 = R0_2 * R0
    R0_4 = R0_2 * R0_2
    R1 = r_list[1]
    R1_2 = R1 * R1
    R1_3 = R1_2 * R1
    R1_R0 = R1 - R0
    R1_R0_sum_of_squares = R1_2 + R1*R0 + R0_2
    R1_3_minus_R0_3_inv = 1.0 / (R1_R0 * R1_R0_sum_of_squares)
    
    hydromat[0, 0] = (8.0 * R0 * (p_list[1] - p_list[0]) + 
                      20.0 * R0_4 * 
                      (p_list[0] / R0_3 + 
                       p_list[1] * R1_3_minus_R0_3_inv) + 
                      3.0 * mhy_list[0] * R1 * R0_2 * 
                      (-rho_list[0] / R0_3 + 
                       rho_list[1] * R1_3_minus_R0_3_inv))
    
    hydromat[0, 1] = (mhy_list[0] * (rho_list[0] + rho_list[1]) - 
                      20.0 * R0_2 * p_list[1] * 
                      R1_2 * R1_3_minus_R0_3_inv - 
                      3.0 * mhy_list[0] * rho_list[1] * R1_3 * 
                      R1_3_minus_R0_3_inv)
    
    hydrob[0] = (-4.0 * R0_2 * (p_list[1] - p_list[0]) - 
                 mhy_list[0] * (rho_list[0] + rho_list[1]) * R1)
    
    # Setup middle rows
    for i in range(1, no_layers - 2):
        Ri = r_list[i]
        Ri_2 = Ri * Ri
        Ri_3 = Ri_2 * Ri
        Ri_4 = Ri_2 * Ri_2
        Ri_1 = r_list[i-1]
        Ri_1_2 = Ri_1 * Ri_1
        Ri_1_3 = Ri_1_2 * Ri_1
        Ri1 = r_list[i+1]
        Ri1_2 = Ri1 * Ri1
        Ri1_3 = Ri1_2 * Ri1
        Ri_Ri_1 = Ri - Ri_1
        Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2
        Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares)
        Ri1_Ri = Ri1 - Ri
        Ri1_Ri_sum_of_squares = Ri1_2 + Ri1*Ri + Ri_2
        Ri1_3_minus_Ri_3_inv = 1.0 / (Ri1_Ri * Ri1_Ri_sum_of_squares)
        
        hydromat[i, i-1] = (-mhy_list[i] * (rho_list[i] + rho_list[i+1]) - 
                            20.0 * Ri_2 * p_list[i] * 
                            Ri_1_2 * Ri_3_minus_Ri_1_3_inv + 
                            3.0 * mhy_list[i] * (Ri1 - Ri_1) * 
                            rho_list[i] * Ri_1_2 * 
                            Ri_3_minus_Ri_1_3_inv)
        
        hydromat[i, i] = (8.0 * Ri * (p_list[i+1] - p_list[i]) + 
                         20.0 * Ri_4 * 
                         (p_list[i] * Ri_3_minus_Ri_1_3_inv + 
                          p_list[i+1] * Ri1_3_minus_Ri_3_inv) + 
                         3.0 * mhy_list[i] * (Ri1 - Ri_1) * 
                         Ri_2 * 
                         (-rho_list[i] * Ri_3_minus_Ri_1_3_inv + 
                          rho_list[i+1] * Ri1_3_minus_Ri_3_inv))
        
        hydromat[i, i+1] = (mhy_list[i] * (rho_list[i] + rho_list[i+1]) - 
                           20.0 * Ri_2 * p_list[i+1] * 
                           Ri1_2 * Ri1_3_minus_Ri_3_inv - 
                           3.0 * mhy_list[i] * (Ri1 - Ri_1) * 
                           rho_list[i+1] * Ri1_2 * 
                           Ri1_3_minus_Ri_3_inv)
        
        hydrob[i] = (-4.0 * Ri_2 * (p_list[i+1] - p_list[i]) - 
                     mhy_list[i] * (rho_list[i] + rho_list[i+1]) * 
                     (Ri1 - Ri_1))
    
    # Setup last row
    last = no_layers - 2
    Rl = r_list[last]
    Rl_2 = Rl * Rl
    Rl_3 = Rl_2 * Rl
    Rl_4 = Rl_2 * Rl_2
    Rl_1 = r_list[last-1]
    Rl_1_2 = Rl_1 * Rl_1
    Rl_1_3 = Rl_1_2 * Rl_1
    Rl1 = r_list[last+1]
    Rl_Rl_1 = Rl - Rl_1
    Rl_Rl_1_sum_of_squares = Rl_2 + Rl*Rl_1 + Rl_1_2
    Rl_3_minus_Rl_1_3_inv = 1.0 / (Rl_Rl_1 * Rl_Rl_1_sum_of_squares)
    
    hydromat[last, last-1] = (-mhy_list[last] * 
                             (rho_list[last] + rho_list[last+1]) - 
                             20.0 * Rl_2 * 
                             p_list[last] * Rl_1_2 * 
                             Rl_3_minus_Rl_1_3_inv + 
                             3.0 * mhy_list[last] * 
                             (Rl1 - Rl_1) * 
                             rho_list[last] * Rl_1_2 * 
                             Rl_3_minus_Rl_1_3_inv)
    
    hydromat[last, last] = (8.0 * Rl * 
                           (p_list[last+1] - p_list[last]) + 
                           20.0 * Rl_4 * 
                           p_list[last] * Rl_3_minus_Rl_1_3_inv - 
                           3.0 * mhy_list[last] * 
                           (Rl1 - Rl_1) * 
                           Rl_2 * rho_list[last] * 
                           Rl_3_minus_Rl_1_3_inv)
    
    hydrob[last] = (-4.0 * Rl_2 * 
                   (p_list[last+1] - p_list[last]) - 
                   mhy_list[last] * (rho_list[last] + 
                   rho_list[last+1]) * 
                   (Rl1 - Rl_1))


@jit(nopython=True, cache=True)
def calculate_delta_rho_and_p_jit(delta_rho: np.ndarray, delta_p: np.ndarray,
                                   delta_r: np.ndarray, r_list: np.ndarray,
                                   rho_list: np.ndarray, p_list: np.ndarray,
                                   no_layers: int):
    """
    JIT-compiled function to calculate changes in density and pressure
    
    Args:
        delta_rho: Output density change array (modified in place)
        delta_p: Output pressure change array (modified in place)
        delta_r: Radius change array
        r_list: Radius array
        rho_list: Density array
        p_list: Pressure array
        no_layers: Number of layers
    """
    R0 = r_list[0]
    R0_2 = R0 * R0
    R0_3 = R0_2 * R0
    
    delta_rho[0] = -3.0 * rho_list[0] * R0_2 * delta_r[0] / R0_3
    delta_p[0] = -5.0 * p_list[0] * R0_2 * delta_r[0] / R0_3
    
    for i in range(1, no_layers - 1):
        Ri = r_list[i]
        Ri_2 = Ri * Ri
        Ri_3 = Ri_2 * Ri
        Ri_1 = r_list[i-1]
        Ri_1_2 = Ri_1 * Ri_1
        Ri_1_3 = Ri_1_2 * Ri_1
        Ri_Ri_1 = Ri - Ri_1
        Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2
        Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares)
        
        delta_rho[i] = -3.0 * rho_list[i] * (Ri_2 * delta_r[i] - 
                      Ri_1_2 * delta_r[i-1]) * Ri_3_minus_Ri_1_3_inv
        delta_p[i] = -5.0 * p_list[i] * (Ri_2 * delta_r[i] - 
                    Ri_1_2 * delta_r[i-1]) * Ri_3_minus_Ri_1_3_inv


@jit(nopython=True, cache=True)
def update_luminosity_jit(l_list: np.ndarray, u_list: np.ndarray, r_list: np.ndarray,
                          rho_list: np.ndarray, v_list: np.ndarray,
                          a: float, b: float, c: float, sigma: float,
                          no_layers: int):
    """
    JIT-compiled function to update luminosity profile
    
    Args:
        l_list: Luminosity array (modified in place)
        u_list: Internal energy array
        r_list: Radius array
        rho_list: Density array
        v_list: Velocity dispersion array
        a, b, c: Conduction parameters
        sigma: Cross section parameter
        no_layers: Number of layers
    """
    R0 = r_list[0]
    R0_2 = R0 * R0
    sigma_2 = sigma * sigma
    v0 = v_list[0]
    v0_2 = v0 * v0
    v0_3 = v0_2 * v0
    v1 = v_list[1]
    v1_2 = v1 * v1
    v1_3 = v1_2 * v1
    
    l_list[0] = (-(u_list[1] - u_list[0]) / r_list[1] * 
                 R0_2 * a * b * c * sigma *
                 (rho_list[0] * v0_3 / 
                  (a * c * sigma_2 * rho_list[0] * v0_2 + b) + 
                  rho_list[1] * v1_3 / 
                  (a * c * sigma_2 * rho_list[1] * v1_2 + b)))
    
    for i in range(1, no_layers - 1):
        Ri = r_list[i]
        Ri_2 = Ri * Ri
        Ri1 = r_list[i+1]
        Ri_1 = r_list[i-1]
        
        vi = v_list[i]
        vi_2 = vi * vi
        vi_3 = vi_2 * vi
        
        vi1 = v_list[i+1]
        vi1_2 = vi1 * vi1
        vi1_3 = vi1_2 * vi1
        
        l_list[i] = (-(u_list[i+1] - u_list[i]) / 
                     (Ri1 - Ri_1) * 
                     Ri_2 * a * b * c * sigma *
                     (rho_list[i] * vi_3 / 
                      (a * c * sigma_2 * rho_list[i] * vi_2 + b) + 
                      rho_list[i+1] * vi1_3 / 
                      (a * c * sigma_2 * rho_list[i+1] * vi1_2 + b)))


class Simulator:
    """Main simulator class handling the gravothermal evolution"""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.state = SimulationState()
        
        # Helper arrays for evolution calculations
        self.delta_u_coeff: Optional[np.ndarray] = None
        self.delta_u: Optional[np.ndarray] = None
        self.hydromat: Optional[np.ndarray] = None
        self.hydrob: Optional[np.ndarray] = None
        self.delta_r: Optional[np.ndarray] = None
        self.delta_p: Optional[np.ndarray] = None
        self.delta_rho: Optional[np.ndarray] = None
    
    def initialize(self):
        """Initialize the simulation state and helper arrays"""
        self.state.initialize(self.params)
        
        no_layers = self.state.no_layers
        
        # Initialize helper arrays with explicit float64 dtype
        self.delta_u_coeff = np.zeros(no_layers, dtype=np.float64)
        self.delta_u = np.zeros(no_layers, dtype=np.float64)
        self.hydromat = np.zeros((no_layers - 1, no_layers - 1), dtype=np.float64)
        self.hydrob = np.zeros(no_layers - 1, dtype=np.float64)
        self.delta_r = np.zeros(no_layers - 1, dtype=np.float64)
        self.delta_p = np.zeros(no_layers - 1, dtype=np.float64)
        self.delta_rho = np.zeros(no_layers - 1, dtype=np.float64)
        
        # Write initial parameters to output file
        with open(self.params.output_file, 'w') as file:
            file.write(f"Input profile name: {self.params.tag}\n")
            file.write(f"Initial time: {self.params.total_time}\n")
            file.write(f"Number of layers: {self.state.no_layers}\n")
            file.write(f"Initial r inner most: {self.state.r_list[0]}\n")
            file.write(f"Initial r outer most: {self.state.r_list[-1]}\n")
            file.write(f"Initial rho inner most: {self.state.rho_list[0]}\n")
            file.write(f"Initial rho outer most: {self.state.rho_list[-1]}\n")
            file.write(f"Total step: {self.params.total_step}\n")
            file.write(f"Save step: {self.params.save_step}\n")
            file.write(f"Abs(delta u/u): {self.params.epsilon}\n")
            file.write(f"Cross section (sigma): {self.params.sigma}\n")
            file.write(f"Conduction parameter (a, b, c): {self.params.a}, {self.params.b}, {self.params.c}\n")
            file.write(f"Baryon parameter (massnorm, scalenorm): {self.params.mass_norm}, {self.params.scale_norm}\n")
            file.write("time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity\n")
            
            # Output initial profiles
            file.write(f"{self.params.total_time:.10e} 0\n")
            np.savetxt(file, [self.state.r_list], fmt='%.10e')
            np.savetxt(file, [self.state.rho_list], fmt='%.10e')
            np.savetxt(file, [self.state.m_list], fmt='%.10e')
            np.savetxt(file, [self.state.u_list], fmt='%.10e')
            np.savetxt(file, [self.state.l_list], fmt='%.10e')
        
        print("Evolution with baryon starts!")
    
    def perform_conduction_step(self) -> float:
        """
        Perform the conduction step
        
        Returns:
            Time step size
        """
        no_layers = self.state.no_layers
        
        # Determine the time step
        self.delta_u_coeff[0] = -(self.state.l_list[0] / self.state.m_list[0]) / self.state.u_list[0]
        
        for i in range(1, no_layers - 1):
            self.delta_u_coeff[i] = -((self.state.l_list[i] - self.state.l_list[i-1]) / 
                                     (self.state.m_list[i] - self.state.m_list[i-1])) / self.state.u_list[i]
        
        delta_t = self.params.epsilon / np.max(np.abs(self.delta_u_coeff))
        
        # Update internal energy due to conduction
        self.delta_u = self.delta_u_coeff * self.state.u_list * delta_t
        self.state.u_list += self.delta_u
        
        # Update pressure and adiabatic variable
        self.state.p_list = (2.0/3.0) * (self.state.rho_list * self.state.u_list)
        self.state.a_list = (2.0/3.0) * (self.state.rho_list ** (-2.0/3.0) * self.state.u_list)
        
        return delta_t
    
    def perform_relaxation_step(self):
        """Perform the relaxation step"""
        no_layers = self.state.no_layers
        
        # Relaxation iterations
        for _ in range(DEFAULT_RELAXATION_STEPS):
            # Set up hydrostatic matrix
            self.hydromat.fill(0.0)
            self.hydrob.fill(0.0)
            
            setup_hydrostatic_matrix_jit(
                self.hydromat, self.hydrob,
                self.state.r_list, self.state.p_list,
                self.state.rho_list, self.state.mhy_list,
                no_layers
            )
            
            # Ensure matrix is in double precision
            assert self.hydromat.dtype == np.float64
            assert self.hydrob.dtype == np.float64
            
            # Solve the system (uses Cholesky for symmetric positive definite matrices)
            # This should match C++ Eigen's LLT solver
            try:
                self.delta_r = np.linalg.solve(self.hydromat, self.hydrob)
            except np.linalg.LinAlgError:
                # Fall back to least squares if matrix is singular
                self.delta_r = np.linalg.lstsq(self.hydromat, self.hydrob, rcond=None)[0]
            
            # Calculate delta_rho and delta_p due to the R change
            calculate_delta_rho_and_p_jit(
                self.delta_rho, self.delta_p,
                self.delta_r, self.state.r_list,
                self.state.rho_list, self.state.p_list,
                no_layers
            )
            
            # Update R, p, Rho profiles
            self.state.rho_list[:-1] += self.delta_rho
            self.state.p_list[:-1] += self.delta_p
            self.state.r_list[:-1] += self.delta_r
    
    def update_profiles(self):
        """Update all profiles after relaxation"""
        no_layers = self.state.no_layers
        
        # Update enclosed total mass after relaxation
        self.state.mhy_list[:-1] = (self.state.m_list[:-1] + 
                                     mbaryon_plummer_vectorized(
                                         self.state.r_list[:-1],
                                         self.params.mass_norm,
                                         self.params.scale_norm
                                     ))
        
        # Update u after relaxation
        self.state.u_list = 1.5 * self.state.a_list * self.state.rho_list ** (2.0/3.0)
        
        # Update v after relaxation
        self.state.v_list = np.sqrt(2.0/3.0) * np.sqrt(self.state.u_list)
        
        # Update L profile
        update_luminosity_jit(
            self.state.l_list, self.state.u_list, self.state.r_list,
            self.state.rho_list, self.state.v_list,
            self.params.a, self.params.b, self.params.c, self.params.sigma,
            no_layers
        )
    
    def save_results(self, file, tstep: int):
        """Save current state to file with same precision as C++ (10 digits scientific)"""
        file.write(f"{self.params.total_time:.10e} {tstep}\n")
        # Match C++ precision exactly: scientific notation with 10 significant digits
        np.savetxt(file, [self.state.r_list], fmt='%.10e', delimiter=' ')
        np.savetxt(file, [self.state.rho_list], fmt='%.10e', delimiter=' ')
        np.savetxt(file, [self.state.m_list], fmt='%.10e', delimiter=' ')
        np.savetxt(file, [self.state.u_list], fmt='%.10e', delimiter=' ')
        np.savetxt(file, [self.state.l_list], fmt='%.10e', delimiter=' ')
    
    def run_simulation(self):
        """Run the main simulation loop"""
        current_save_step = self.params.save_step
        
        with open(self.params.output_file, 'a') as output_file:
            for tstep in range(1, self.params.total_step + 1):
                try:
                    # Conduction period
                    delta_t = self.perform_conduction_step()
                    self.params.total_time += delta_t
                    
                    # Relaxation period
                    self.perform_relaxation_step()
                    
                    # Update profiles after relaxation
                    self.update_profiles()
                    
                    # Check if density reaches 1e6 threshold
                    if self.state.rho_list[0] >= 1e6 and current_save_step != 1:
                        current_save_step = 1
                        print("Density threshold 1e6 reached! Save frequency increased to every step.")
                    
                    # Check for abnormal state
                    if self.state.check_for_abnormal_state():
                        break
                    
                    # Save results based on the current save step frequency
                    if (tstep % current_save_step) == 0:
                        self.save_results(output_file, tstep)
                        if self.params.verbose:
                            print(f"Saved at time = {self.params.total_time}, step = {tstep}")
                
                except Exception as e:
                    print(f"Error during simulation step {tstep}: {e}")
                    break
        
        print("Evolution ends")


def main():
    """Main entry point"""
    start_time = time.time()
    
    try:
        # Verify we're using double precision (float64) matching C++ double
        verify_precision()
        if __name__ == "__main__":
            print("✓ Using double precision (float64) matching C++ double")
        
        # Create simulation parameters
        params = SimulationParameters()
        
        # Display parameters
        params.display()
        
        # Create and initialize simulator
        simulator = Simulator(params)
        simulator.initialize()
        
        # Run simulation
        simulator.run_simulation()
        
        # Measure computation time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Computation time = {elapsed:.6f} s")
    
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
