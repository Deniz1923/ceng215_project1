# Generic RLC circuit solver and plotter
# Usage: run this script or import functions `solve_rlc` / `plot_rlc_response`

import argparse
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def rlc_ode(t: float, y: np.ndarray, R: float, L: float, C: float, Vin: Callable[[float], float]):
	"""
	State equations for a series RLC circuit.
	y[0] = q (charge), y[1] = i (current)
	dq/dt = i
	di/dt = (Vin(t) - R*i - q/C) / L
	"""
	q, i = y
	dqdt = i
	didt = (Vin(t) - R * i - q / C) / L
	return [dqdt, didt]


def solve_rlc(R: float, L: float, C: float, Vin: Callable[[float], float],
			  t_span: Tuple[float, float], y0=(0.0, 0.0), t_eval=None):
	"""Solve the series RLC ODE.

	Returns the solution object from `scipy.integrate.solve_ivp`.
	"""
	if t_eval is None:
		t_eval = np.linspace(t_span[0], t_span[1], 2000)

	sol = solve_ivp(rlc_ode, t_span, y0, t_eval=t_eval, args=(R, L, C, Vin), rtol=1e-8)
	return sol


def default_vin_step(amplitude=1.0, t0=0.0):
	return lambda t: amplitude if t >= t0 else 0.0


def default_vin_sine(amplitude=1.0, freq=50.0, phase=0.0):
	return lambda t: amplitude * np.sin(2 * np.pi * freq * t + phase)


def plot_rlc_response(sol, R: float, L: float, C: float, Vin_func: Callable[[float], float],
					  save_path: str = None, show: bool = True):
	t = sol.t
	q = sol.y[0]
	i = sol.y[1]

	# Component voltages
	vR = R * i
	vL = L * np.gradient(i, t)
	vC = q / C

	fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
	axs[0].plot(t, [Vin_func(tt) for tt in t], label='Vin')
	axs[0].plot(t, vR + vL + vC, '--', label='Sum (should match Vin)')
	axs[0].set_ylabel('Voltage (V)')
	axs[0].legend()

	axs[1].plot(t, vR, label='vR')
	axs[1].plot(t, vL, label='vL')
	axs[1].plot(t, vC, label='vC')
	axs[1].set_ylabel('Voltage (V)')
	axs[1].legend()

	axs[2].plot(t, i, color='C3', label='Current i(t)')
	axs[2].set_xlabel('Time (s)')
	axs[2].set_ylabel('Current (A)')
	axs[2].legend()

	fig.tight_layout()
	if save_path:
		fig.savefig(save_path, dpi=150)
		print(f'Saved plot to {save_path}')
	if show:
		plt.show()
	plt.close(fig)


def parse_args():
	p = argparse.ArgumentParser(description='Generic series RLC solver and plotter')
	p.add_argument('--R', type=float, default=10.0, help='Resistance in ohms')
	p.add_argument('--L', type=float, default=0.1, help='Inductance in henrys')
	p.add_argument('--C', type=float, default=1e-6, help='Capacitance in farads')
	p.add_argument('--input', choices=['step', 'sine'], default='step', help='Input type')
	p.add_argument('--amp', type=float, default=1.0, help='Input amplitude')
	p.add_argument('--freq', type=float, default=1000.0, help='Sine frequency (Hz)')
	p.add_argument('--duration', type=float, default=0.01, help='Simulation duration (s)')
	p.add_argument('--save', type=str, default='rlc_response.png', help='Path to save the plot')
	return p.parse_args()


def main():
	args = parse_args()

	R, L, C = args.R, args.L, args.C

	if args.input == 'step':
		Vin = default_vin_step(amplitude=args.amp, t0=0.0)
	else:
		Vin = default_vin_sine(amplitude=args.amp, freq=args.freq)

	t_span = (0.0, args.duration)
	t_eval = np.linspace(t_span[0], t_span[1], 5000)

	sol = solve_rlc(R, L, C, Vin, t_span, y0=(0.0, 0.0), t_eval=t_eval)

	plot_rlc_response(sol, R, L, C, Vin, save_path=args.save, show=True)


if __name__ == '__main__':
	main()

