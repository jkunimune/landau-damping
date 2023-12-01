from matplotlib import pyplot as plt
from numpy import linspace, random, pi, zeros, histogram2d, hypot, concatenate, sin, stack, ravel
from scipy import integrate

k = 3*pi  # three eyes in frame at any time
ω = 4*pi  # two full oscillations in a 1s video
vth = 0.5*ω/k  # ensure a high gradient at the wave velocity


def main():

	x_grid = linspace(-1, 1, 301)  # normalized spacial coordinates
	v_grid = linspace(-0.4*ω/k, 1.8*ω/k, 201)  # set velocity bounds to see wave velocityu

	frame_rate = 24

	g0 = .1

	v0 = random.normal(0, vth, 100000)  # maxwellian inital distribution
	v0 = v0[(v0 > v_grid[0]*ω/k) & (v0 < v_grid[-1] + 0.1*ω/k)]  # exclude particles off screen
	x0 = random.uniform(-1, 1, v0.size)  # randomize position as well

	# solve with respect to time
	def derivative(t, state):
		x = state[0::2]
		v = state[1::2]
		dxdt = v
		dvdt = g0*sin(k*x - ω*t)
		return ravel(stack([dxdt, dvdt], axis=1))
	solution = integrate.solve_ivp(derivative, t_span=(0, 4),
	                               t_eval=linspace(0, 4, 4*frame_rate, endpoint=False),
	                               y0=ravel(stack([x0, v0], axis=1)))
	t = solution.t  # type: ignore
	x = solution.y[0::2, :]  # type: ignore
	v = solution.y[1::2, :]  # type: ignore

	plot_phase_space(x_grid, v_grid, t, x, v)
	plt.show()


def plot_phase_space(x_grid, v_grid, t, x, v):
	plt.figure()
	for i in range(len(t)):
		r_particle = (x_grid[1] - x_grid[0])*2.0
		histogram = zeros((x_grid.size - 1, v_grid.size - 1))
		for dx in linspace(-r_particle, r_particle, 9):
			for dy in linspace(-r_particle, r_particle, 9):
				if hypot(dx, dy) < r_particle:
					dv = dy/(x_grid[1] - x_grid[0])*(v_grid[1] - v_grid[0])
					histogram += histogram2d(x[:, i] + dx, v[:, i] + dv, bins=(x_grid, v_grid))[0]
		plt.imshow(histogram.transpose(), extent=(x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]),
		           vmin=0, vmax=180, aspect="auto",
		           origin="lower", cmap="bone_r")
		plt.pause(0.05)
	plt.show()


if __name__ == "__main__":
	main()
