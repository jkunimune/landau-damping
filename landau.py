import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace, random, pi, zeros, histogram2d, hypot, sin, stack, ravel, cos, histogram, repeat
from scipy import integrate

from colormap import colormap

k = 3*pi  # three eyes in frame at any time
ω = 4*pi  # two full oscillations in a 1s video
vth = 0.5*ω/k  # ensure a high gradient at the wave velocity
g0 = .1


def main():

	x_grid = linspace(-1, 1, 301)  # normalized spacial coordinates
	v_grid = linspace(-0.4*ω/k, 1.8*ω/k, 201)  # set velocity bounds to see wave velocityu

	frame_rate = 24


	v0 = random.normal(0, vth, 100000)  # maxwellian inital distribution
	v0 = v0[(v0 > v_grid[0]*ω/k) & (v0 < v_grid[-1] + 0.1*ω/k)]  # exclude particles off screen
	x0 = random.uniform(-1, 1, v0.size)  # randomize position as well

	# solve with respect to time
	def derivative(t, state):
		x = state[0::2]
		v = state[1::2]
		dxdt = v
		dvdt = g0*cos(k*x - ω*t)
		return ravel(stack([dxdt, dvdt], axis=1))
	solution = integrate.solve_ivp(derivative, t_span=(0, 4),
	                               t_eval=linspace(0, 4, 4*frame_rate, endpoint=False),
	                               y0=ravel(stack([x0, v0], axis=1)))
	t = solution.t  # type: ignore
	x = solution.y[0::2, :]  # type: ignore
	v = solution.y[1::2, :]  # type: ignore

	# plot it
	plot_phase_space(x_grid, v_grid, t, x, v)
	plt.show()


def plot_phase_space(x_grid, v_grid, t, x, v):
	fig, ((ax_E, space), (ax_image, ax_v)) = plt.subplots(
		nrows=2, ncols=2, facecolor="none", sharex="col", sharey="row",
		gridspec_kw=dict(
			hspace=0, wspace=0, width_ratios=[4, 1], height_ratios=[1, 4])
	)
	ax_V = ax_E.twinx()

	space.axis("off")

	for i in range(len(t)):
		# plot the electric field and potential as functions of space
		ax_E.clear()
		ax_E.set_yticks([])
		ax_E.set_ylabel("Field", color="#672392")
		ax_E.plot(x_grid, g0*cos(k*x_grid - ω*t[i]), color="#672392", zorder=20)
		ax_V.clear()
		ax_V.set_yticks([])
		ax_V.yaxis.set_label_position("right")
		ax_V.set_ylabel("Potential", color="#9a4504", rotation=-90, labelpad=11)
		ax_V.plot(x_grid, g0/k*sin(k*x_grid - ω*t[i]), color="#e1762b", linestyle="dotted", zorder=10)
		ax_E.set_zorder(ax_V.get_zorder()+1)
		ax_E.set_frame_on(False)

		# plot the distribution function as a function of velocity (space-averaged)
		ax_v.clear()
		ax_v.set_xticks([])
		ax_v.set_xlabel("Distribution", color="#215772")
		f_v, v_bins = histogram(v[:, i], v_grid[0::3])
		ax_v.fill_betweenx(repeat(v_bins, 2)[1:-1], 0, repeat(f_v, 2), color="#356884")
		ax_v.set_xlim(0, v[:, i].size*3.2e-2)

		r_particle = (x_grid[1] - x_grid[0])*2.0
		image = zeros((x_grid.size - 1, v_grid.size - 1))
		for dx in linspace(-r_particle, r_particle, 9):
			for dy in linspace(-r_particle, r_particle, 9):
				if hypot(dx, dy) < r_particle:
					dv = dy/(x_grid[1] - x_grid[0])*(v_grid[1] - v_grid[0])
					image += histogram2d(x[:, i] + dx, v[:, i] + dv, bins=(x_grid, v_grid))[0]
		ax_image.clear()
		ax_image.set_xlabel("Position")
		ax_image.set_ylabel("Velocity")
		ax_image.imshow(image.transpose(), extent=(x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]),
		                vmin=0, vmax=180, aspect="auto",
		                origin="lower", cmap=colormap)
		plt.tight_layout()
		plt.pause(0.05)
	plt.show()


if __name__ == "__main__":
	main()
