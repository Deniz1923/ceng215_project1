import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List

# ==========================================
# 1. THE PHYSICS ENGINE (Differential Eq)
# ==========================================
def rlc_ode(t: float, state: List[float], R: float, L: float, C: float, Vin_func: Callable[[float], float]):
    """
    This function defines the 'rules' of the circuit for the solver.
    It returns how charge and current change at every instant.
    
    Args:
        t: Current time (seconds)
        state: A list [charge (q), current (i)]
        R, L, C: Circuit component values
        Vin_func: A function that returns Input Voltage at time t
        
    Returns:
        [dq/dt, di/dt]: The rates of change for charge and current.
    """
    # Unpack the current state
    q = state[0]  # Charge on capacitor (Coulombs)
    i = state[1]  # Current in loop (Amps)

    # PHYSICS EXPLANATION:
    # 1. The rate of change of Charge (q) is exactly Current (i).
    dq_dt = i
    
    # 2. To find the rate of change of Current (di/dt), we rearrange KVL:
    #    Vin = V_R + V_L + V_C
    #    Vin = (R * i) + (L * di/dt) + (q / C)
    #    
    #    Solving for di/dt:
    #    L * di/dt = Vin - R*i - q/C
    #    di/dt = (Vin - R*i - q/C) / L
    
    current_voltage = Vin_func(t)
    di_dt = (current_voltage - (R * i) - (q / C)) / L
    
    return [dq_dt, di_dt]

# ==========================================
# 2. THE SOLVER
# ==========================================
def solve_circuit(R, L, C, Vin_func, t_duration=0.1):
    """
    Sets up the time steps and runs the numerical solver (Runge-Kutta).
    """
    # Simulation settings
    t_span = (0.0, t_duration)           # Start to End time
    t_eval = np.linspace(0, t_duration, 5000) # Points to record (high res for smooth plots)
    initial_state = [0.0, 0.0]           # Start with 0 Charge and 0 Current

    # Call Scipy's solver
    # This automatically steps through time, calling 'rlc_ode' at every step
    solution = solve_ivp(
        rlc_ode, 
        t_span, 
        initial_state, 
        t_eval=t_eval, 
        args=(R, L, C, Vin_func),
        method='RK45', # Standard reliable solver method
        rtol=1e-9      # High precision
    )
    return solution

# ==========================================
# 3. HELPER: INPUT SIGNALS
# ==========================================
def get_step_input(amplitude=5.0):
    """Returns a function that instantly turns on voltage to 'amplitude'."""
    return lambda t: amplitude if t > 0 else 0.0

# ==========================================
# 4. THE VISUALIZER
# ==========================================
def plot_results(sol, R, L, C, Vin_func):
    """
    Calculates voltages from the solver results and plots them cleanly.
    """
    time = sol.t
    q = sol.y[0] # Charge over time
    i = sol.y[1] # Current over time

    # Calculate Voltages across components
    # V_R = Ohm's Law
    v_resistor = R * i 
    
    # V_C = Q / C
    v_capacitor = q / C 
    
    # V_L: We calculate this Algebraically using KVL to avoid noisy derivatives.
    # V_L = Source - V_R - V_C
    # We use a list comprehension to calculate this for every time point
    v_source_arr = np.array([Vin_func(t) for t in time])
    v_inductor = v_source_arr - v_resistor - v_capacitor

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Voltages
    ax1.set_title(rf"Series RLC Response (R={R}$\Omega$, L={L}H, C={C}F)", fontsize=14)
    ax1.plot(time, v_source_arr, 'k--', label='$V_{in}$ (Source)', alpha=0.6)
    ax1.plot(time, v_resistor, 'g-', label='$V_R$ (Resistor)', linewidth=1.5)
    ax1.plot(time, v_capacitor, 'b-', label='$V_C$ (Capacitor)', linewidth=1.5)
    ax1.plot(time, v_inductor, 'r-', label='$V_L$ (Inductor)', linewidth=1.5)
    ax1.set_ylabel("Voltage (Volts)")
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', alpha=0.3)

    # Plot 2: Current
    ax2.plot(time, i, 'm-', label='Current $i(t)$', linewidth=2)
    ax2.set_ylabel("Current (Amps)")
    ax2.set_xlabel("Time (seconds)")
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CIRCUIT VALUES ---
    # Try changing these to see different behaviors!
    
    # UNDERDAMPED CASE (Bouncy, oscillates)
    R_val = 2.0      # Low resistance = lots of oscillation
    L_val = 0.1      # Inductance
    C_val = 100e-6   # 100 microFarads
    
    # Setup Input: 10 Volt Step (switch turns on at t=0)
    voltage_source = get_step_input(amplitude=10.0)

    # Solve
    print("Solving RLC Circuit...")
    sol = solve_circuit(R_val, L_val, C_val, voltage_source, t_duration=0.15)
    
    # Plot
    print("Plotting results...")
    plot_results(sol, R_val, L_val, C_val, voltage_source)