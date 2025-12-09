import numpy as np
import matplotlib.pyplot as plt

class ParallelRLCSolver:
    def __init__(self, R, L, C, initial_vc=0.0, initial_il=0.0):
        """
        Initialize the Parallel RLC System.
        
        State variables:
        x[0] = v_C (Capacitor Voltage)
        x[1] = i_L (Inductor Current)
        
        Energy storage definitions based on.
        """
        self.R = R
        self.L = L
        self.C = C
        self.x0 = np.array([initial_vc, initial_il])
        
        # Calculate damping factor alpha and resonant freq omega_0 for reference
        self.alpha = 1 / (2 * R * C)
        self.omega0 = 1 / np.sqrt(L * C)
        print(f"System Parameters: alpha={self.alpha:.2e}, w0={self.omega0:.2e}")

    def state_derivatives(self, t, x, u):
        """
        Computes x_dot = f(t, x, u)
        
        Equations derived from nodal analysis (KCL):
        dv_C/dt = (1/C) * (i_source - v_C/R - i_L)
        di_L/dt = (1/L) * v_C
        """
        v_C = x[0]
        i_L = x[1]
        i_source = u
        
        dv_C_dt = (1.0 / self.C) * (i_source - (v_C / self.R) - i_L)
        di_L_dt = (1.0 / self.L) * v_C
        
        return np.array([dv_C_dt, di_L_dt])

    def simulate(self, T_end, h, source_func):
        """
        Runs the simulation using Forward Euler.
        
        Args:
            T_end: Total simulation time
            h: Time step size
            source_func: Function taking time 't' and returning source current i_s(t)
        """
        # Time grid creation [cite: 27]
        N = int(np.ceil(T_end / h)) + 1
        t = np.linspace(0.0, N * h, N)
        
        # Initialize state matrix (rows=time, cols=states)
        x = np.zeros((N, 2))
        x[0] = self.x0
        
        u_rec = np.zeros(N) # Record input for plotting
        
        # Euler Loop 
        for k in range(N - 1):
            current_time = t[k]
            current_x = x[k]
            
            # Evaluate input source at current time
            u = source_func(current_time)
            u_rec[k] = u
            
            # Calculate derivatives
            f = self.state_derivatives(current_time, current_x, u)
            
            # Update state: x[k+1] = x[k] + h * f
            x[k+1] = current_x + h * f
            
        u_rec[-1] = source_func(t[-1]) # Last point
        return t, x, u_rec

# --- Usage Example ---

# 1. Define Circuit Parameters
# Using values that create an underdamped response
R_val = 1000.0  # Ohms
L_val = 100e-3  # Henries (100mH)
C_val = 1e-6    # Farads (1uF)

solver = ParallelRLCSolver(R_val, L_val, C_val)

# 2. Define Simulation Parameters
# Stability: h should be small relative to time constants 
# Using h approx 1% of the period 1/f0
f0 = solver.omega0 / (2*np.pi)
period = 1/f0
T_total = 10 * period
h_step = period / 1000 

print(f"Simulation: T_end={T_total:.4f}s, h={h_step:.6f}s")

# 3. Define Input Source (Step Current)
def step_current(t):
    return 0.01 if t > 0 else 0.0 # 10mA step

# 4. Run Simulation
t_data, x_data, u_data = solver.simulate(T_total, h_step, step_current)

# Extract states
v_C = x_data[:, 0]
i_L = x_data[:, 1]

# 5. Plot Results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_data, u_data * 1000, 'g--', label='Source Current $i_s(t)$ (mA)')
plt.plot(t_data, i_L * 1000, 'b', label='Inductor Current $i_L(t)$ (mA)')
plt.ylabel('Current (mA)')
plt.title('Parallel RLC Step Response (Forward Euler)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(t_data, v_C, 'r', label='Capacitor Voltage $v_C(t)$ (V)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()