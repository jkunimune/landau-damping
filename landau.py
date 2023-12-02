from os import makedirs

from imageio.v2 import mimsave
from imageio.v3 import imread
from matplotlib import pyplot as plt
from numpy import linspace, random, pi, zeros, histogram2d, hypot, sin, stack, ravel, cos, histogram, repeat, \
	zeros_like, sqrt, meshgrid, arange, concatenate, full, size, where
from numpy.typing import NDArray
from scipy import integrate

from colormap import colormap

k = 3*pi  # three eyes in frame at any time
ω = 4*pi  # two full oscillations in 1s
vth = 0.7*ω/k  # ensure a high gradient at the wave velocity
g0 = .4


def main():

	random.seed(0)
	makedirs("output", exist_ok=True)

	x_grid = linspace(-1, 1, 301)  # normalized spacial coordinates
	v_grid = linspace(-0.4*ω/k, 2.0*ω/k, 201)  # set velocity bounds to see wave velocityu

	frame_rate = 24
	duration = 5

	v0 = random.normal(0, vth, 80000)  # maxwellian inital distribution
	v0 = v0[(v0 > v_grid[0]*ω/k) & (v0 < v_grid[-1] + 0.1*ω/k)]  # exclude particles off screen
	x0 = random.uniform(-1, 1, v0.size)  # randomize position as well

	for field_on in [True, False]:

		# solve with respect to time
		def derivative(t, state):
			x = state[0::2]
			v = state[1::2]
			dxdt = v
			dvdt = g0*sin(k*x - ω*t) if field_on else zeros_like(v)
			return ravel(stack([dxdt, dvdt], axis=1))
		solution = integrate.solve_ivp(
			derivative, t_span=(0, duration),
			t_eval=linspace(0, duration, duration*frame_rate, endpoint=False),
			y0=ravel(stack([x0, v0], axis=1)))
		t = solution.t  # type: ignore
		x = solution.y[0::2, :]  # type: ignore
		v = solution.y[1::2, :]  # type: ignore

		for wave_frame in [True, False]:
			for trajectories in [True, False]:

				if wave_frame and not field_on:
					continue  # don't do the wave frame unless there's a wave
				if trajectories and not wave_frame:
					continue  # trajectories are meaningless unless the field is static

				# choose the filename
				filename = "output/distribution"
				if wave_frame:
					filename += "_stationary"
				if field_on:
					filename += "_wave"
				else:
					filename += "_ballistic"
				if trajectories:
					filename += "_with_trajectories"

				# plot it all
				plot_phase_space(x_grid, v_grid, t, x, v, field_on, wave_frame, trajectories,
				                 filename + "_at_t{:03d}.png")

				# combine the images into a single animated image
				make_gif(filename, len(t), frame_rate)

				plt.show()


def plot_phase_space(x_grid_initial: NDArray[float], v_grid: NDArray[float], t: NDArray[float],
                     x: NDArray[float], v: NDArray[float],
                     field_on: bool, wave_frame: bool, trajectories: bool,
                     filename_format: str):
	fig, ((ax_V, space), (ax_image, ax_v)) = plt.subplots(
		nrows=2, ncols=2, facecolor="none", sharex="col", sharey="row",
		gridspec_kw=dict(
			hspace=0, wspace=0, width_ratios=[5, 1], height_ratios=[1, 5])
	)
	ax_E = ax_V.twinx()

	space.axis("off")

	for i in range(len(t)):
		# move with the wave (or not)
		x_grid = x_grid_initial + ω/k*t[i] if wave_frame else x_grid_initial

		# plot the electric field and potential as functions of space
		ax_E.clear()
		ax_E.set_yticks([])
		ax_E.set_ylabel("Field", color="#672392")
		ax_E.plot(x_grid, g0*sin(k*x_grid - ω*t[i]) if field_on else zeros_like(x_grid),
		          color="#672392", zorder=20)
		ax_V.clear()
		ax_V.set_yticks([])
		ax_V.yaxis.set_label_position("right")
		ax_V.set_ylabel("Potential", color="#9a4504", rotation=-90, labelpad=11)
		ax_V.plot(x_grid, g0/k*cos(k*x_grid - ω*t[i]) if field_on else zeros_like(x_grid),
		          color="#e1762b", linestyle="dotted", zorder=10)

		# plot the distribution function as a function of velocity (space-averaged)
		ax_v.clear()
		ax_v.set_xticks([])
		ax_v.set_xlabel("Distribution", color="#215772")
		f_v, v_bins = histogram(v[:, i], v_grid[0::4])
		ax_v.fill_betweenx(repeat(v_bins, 2)[1:-1], 0, repeat(f_v, 2), color="#356884")
		ax_v.set_xlim(0, v[:, i].size*3.8e-2)

		r_particle = (x_grid[1] - x_grid[0])*2.0
		image = zeros((x_grid.size - 1, v_grid.size - 1))
		for dx in linspace(-r_particle, r_particle, 13):
			for dy in linspace(-r_particle, r_particle, 13):
				if hypot(dx, dy) < r_particle*1.1:
					dv = dy/(x_grid[1] - x_grid[0])*(v_grid[1] - v_grid[0])
					image += histogram2d(periodicize(x[:, i] + dx, x_grid[0], x_grid[-1]),
					                     v[:, i] + dv, bins=(x_grid, v_grid))[0]
		ax_image.clear()
		ax_image.set_xlabel("Position")
		ax_image.set_ylabel("Velocity")
		ax_image.imshow(image.transpose(), extent=(x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]),
		                vmin=0, vmax=450 if trajectories else 300,
		                cmap=colormap, aspect="auto", origin="lower")

		if trajectories:
			X_grid, V_grid = meshgrid(x_grid, v_grid)
			v_plot = arange(1/5, 10, 2/5)*2*sqrt(g0/k)
			trajectory_type = concatenate([[0, 0, 1], full(size(v_plot) - 3, 2)])  # 0: trapped, 1: separatrix, 2: passing
			ax_image.contour(x_grid, v_grid,
			                 sqrt((V_grid - ω/k)**2 + 2*g0/k*(cos(k*X_grid - ω*t[i]) + 1)),
			                 levels=v_plot, linewidths=where(trajectory_type == 1, 1.4, 0.7),
			                 colors="k")

		fig.tight_layout()
		fig.savefig(filename_format.format(i), dpi=150)
		plt.pause(0.05)
	plt.close(fig)


def periodicize(x: NDArray[float], minimum: float, maximum: float) -> NDArray[float]:
	return minimum + (x - minimum) % (maximum - minimum)


def make_gif(base_filename: str, num_frames: int, frame_rate: float):
	frames = []
	for i in range(num_frames):
		frames.append(imread(f"{base_filename}_at_t{i:03d}.png"))
	mimsave(f"{base_filename}.gif", frames, fps=frame_rate)


if __name__ == "__main__":
	main()
