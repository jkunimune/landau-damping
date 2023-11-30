from matplotlib import pyplot as plt
from numpy import linspace, random, pi, zeros, histogram2d, hypot, concatenate, sin

k = 7  # a bit more than two eyes in [-1, 1] at any time
ω = 4*pi  # two full oscillations in a 1s video
vth = 0.5*ω/k  # ensure a high gradient at the wave velocity


def main():

	x_grid = linspace(-1, 1, 301)  # normalized spacial coordinates
	v_grid = linspace(-0.4*ω/k, 1.8*ω/k, 201)  # set velocity bounds to see wave velocity

	frame_rate = 24

	E0 = .1

	v0 = random.normal(0, vth, 100000)  # maxwellian inital distribution
	v0 = v0[(v0 > v_grid[0]*ω/k) & (v0 < v_grid[-1] + 0.1*ω/k)]  # exclude particles off screen
	x0 = random.uniform(-1, 1, v0.size)  # randomize position as well

	plot_phase_space(x_grid, v_grid, x0, v0)
	plt.show()


def plot_phase_space(x_grid, v_grid, x, v):
	r_particle = (x_grid[1] - x_grid[0])*2.0
	histogram = zeros((x_grid.size - 1, v_grid.size - 1))
	for dx in linspace(-r_particle, r_particle, 9):
		for dy in linspace(-r_particle, r_particle, 9):
			if hypot(dx, dy) < r_particle:
				dv = dy/(x_grid[1] - x_grid[0])*(v_grid[1] - v_grid[0])
				histogram += histogram2d(x + dx, v + dv, bins=(x_grid, v_grid))[0]
	plt.imshow(histogram.transpose(), extent=(x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]),
	           vmin=0, vmax=180, aspect="auto",
	           origin="lower", cmap="bone_r")


if __name__ == "__main__":
	main()
