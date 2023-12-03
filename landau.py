import logging
from os import makedirs

import numpy as np
from imageio.v2 import mimsave
from imageio.v3 import imread
from matplotlib import pyplot as plt, ticker
from numpy import linspace, random, pi, zeros, histogram2d, hypot, sin, stack, ravel, cos, histogram, repeat, \
	zeros_like, sqrt, meshgrid, arange, concatenate, full, size, where, diff, exp, newaxis, uint8, shape, floor
from numpy.typing import NDArray
from scipy import integrate
from scipy.special import erf

from colormap import colormap

plt.rc("font", size=12)

k = 2*pi  # three full eyes in the simulation domain
ω = 4*pi  # two full oscillations in 1s
g0 = .8  # wave amplitude
t_on = 0.5  # amount of time before wave appears
Δt_on = 0.05  # amount of time it takes wave to ramp up

v_thermal = 0.8*ω/k  # ensure a high gradient at the wave velocity

x_grid = linspace(-1.5, 1.5, 361)  # normalized spacial coordinates
v_grid = linspace(-0.6*ω/k, 2.1*ω/k, 201)  # set velocity bounds to see wave velocity

num_samples = 1_000_000
frame_rate = 30
duration = Δt_on + 8.0


def main():

	logging.info("begin.")

	random.seed(0)
	makedirs("output", exist_ok=True)

	v0 = random.normal(0, v_thermal, num_samples)  # maxwellian inital distribution
	v0 = v0[(v0 > v_grid[0] - 0.1*ω/k) & (v0 < v_grid[-1] + 0.1*ω/k)]  # exclude particles off screen
	x0 = random.uniform(x_grid[0], x_grid[-1], v0.size)  # randomize position as well

	for field_on in [True, False]:

		# solve with respect to time
		def derivative(t, state):
			x = state[0::2]
			v = state[1::2]
			dxdt = v
			dvdt = g0*(1 + erf((t - t_on)/Δt_on))/2*sin(k*x - ω*t) if field_on else zeros_like(v)
			return ravel(stack([dxdt, dvdt], axis=1))
		t = linspace(0, duration, round(duration*frame_rate), endpoint=False)
		solution = integrate.solve_ivp(
			derivative, t_span=(t[0], t[-1]), t_eval=t,
			y0=ravel(stack([x0, v0], axis=1)))
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
				logging.info(f"generating {filename}.gif...")

				# plot it all
				plot_phase_space(x_grid, v_grid, t, x, v, field_on, wave_frame, trajectories,
				                 filename + "_at_t{:03d}.png")

				# combine the images into a single animated image
				make_gif(filename, len(t), frame_rate)

	logging.info("done!")


def plot_phase_space(x_grid_initial: NDArray[float], v_grid: NDArray[float], t: NDArray[float],
                     x: NDArray[float], v: NDArray[float],
                     field_on: bool, wave_frame: bool, trajectories: bool,
                     filename_format: str):
	fig, ((ax_V, textbox), (ax_image, ax_v)) = plt.subplots(
		nrows=2, ncols=2, facecolor="none", sharex="col", sharey="row",
		gridspec_kw=dict(
			left=.095, right=.992, bottom=.100, top=.989,
			hspace=0, wspace=0, width_ratios=[5, 1], height_ratios=[1, 4])
	)
	ax_E = ax_V.twinx()
	ax_image.set_zorder(10)

	for i in range(len(t)):
		# move with the wave (or not)
		x_grid = x_grid_initial + ω/k*t[i] if wave_frame else x_grid_initial

		E = g0*(1 + erf((t[i] - t_on)/Δt_on))/2*sin(k*x_grid - ω*t[i])
		ф = g0*(1 + erf((t[i] - t_on)/Δt_on))/2*cos(k*x_grid - ω*t[i])/k

		# show the current time
		textbox.clear()
		textbox.axis("off")
		textbox.text(.23, .95, f"$t$ = {floor(t[i]*10)/10:3.1f} s",
		             horizontalalignment="left", verticalalignment="top",
		             transform=textbox.transAxes)

		# plot the electric field and potential as functions of space
		ax_E.clear()
		ax_E.set_yticks([])
		ax_E.set_ylabel("Field", color="#672392")
		ax_E.set_ylim(-1.1*g0, 1.1*g0)
		ax_E.plot(x_grid, E if field_on else zeros_like(x_grid),
		          color="#672392", linewidth=1.4, zorder=20)
		ax_V.clear()
		ax_V.set_yticks([])
		ax_V.set_ylabel("Potential", color="#ce661a", labelpad=17)
		ax_V.set_ylim(-1.4*g0/k, 1.4*g0/k)
		ax_V.plot(x_grid, ф if field_on else zeros_like(x_grid),
		          color="#e1762b", linewidth=1.4, linestyle="dotted", zorder=10)

		# plot the distribution function as a function of velocity (space-averaged)
		ax_v.clear()
		ax_v.set_xticks([])
		ax_v.set_xlabel("Distribution", color="#215772")
		f_v, v_bins = histogram(v[:, i], v_grid[0::2])
		ax_v.fill_betweenx(repeat(v_bins, 2)[1:-1], 0, repeat(f_v/diff(v_bins), 2), color="#61909d")
		ax_v.plot(num_samples/sqrt(2*pi)/v_thermal*exp(-(v_bins/v_thermal)**2/2), v_bins,
		          color="k", linewidth=1.0, linestyle="dashed")
		ax_v.set_xlim(0, 0.43*num_samples/v_thermal)

		# plot the 2D distribution function
		r_particle = (x_grid[1] - x_grid[0])*1.5
		image = zeros((x_grid.size - 1, v_grid.size - 1))
		for dx in linspace(-r_particle, r_particle, 7):
			for dy in linspace(-r_particle, r_particle, 7):
				if hypot(dx, dy) < r_particle*1.1:
					dv = dy/(x_grid[1] - x_grid[0])*(v_grid[1] - v_grid[0])
					image += histogram2d(periodicize(x[:, i] + dx, x_grid[0], x_grid[-1]),
					                     v[:, i] + dv, bins=(x_grid, v_grid))[0]
		ax_image.clear()
		ax_image.set_xlabel("Position")
		ax_image.set_ylabel("Velocity", labelpad=-6 if field_on else 0)
		lightening_modifier = 1.5 if trajectories else 1.0
		ax_image.imshow(image.transpose(), extent=(x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]),
		                vmin=0, vmax=num_samples*7.5e-4*lightening_modifier,
		                cmap=colormap, aspect="auto", origin="lower")
		ax_image.set_xlim(x_grid[0] + .21, x_grid[-1] - .21)
		ax_image.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
		if field_on:
			ax_image.yaxis.set_major_locator(ticker.MultipleLocator(0.5*ω/k))
			ax_image.yaxis.set_major_formatter(lambda v, _: format_as_fraction(v/(ω/k), "ω", "k"))
			ax_image.tick_params(axis="y", which="major", labelsize=15)
		else:
			ax_image.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

		# plot the phase-space trajectories, if desired
		if trajectories:
			X_grid, V_grid = meshgrid(x_grid, v_grid)
			V_max_grid = sqrt((V_grid - ω/k)**2 + 2*(ф - np.min(ф)))
			v_max_separatrix = 2*sqrt(g0/k*(1 + erf((t[i] - t_on)/Δt_on))/2)
			v_max_contours = arange(-4/5, 9, 2/5)*2*sqrt(g0/k) + v_max_separatrix
			if t[i] >= t_on + Δt_on:
				trajectory_type = concatenate([[0, 0, 1], full(size(v_max_contours) - 3, 2)])  # 0: trapped, 1: separatrix, 2: passing
			else:
				trajectory_type = full(size(v_max_contours), 2)
			ax_image.contour(x_grid, v_grid, V_max_grid,
			                 levels=v_max_contours, linewidths=where(trajectory_type == 1, 1.4, 0.7),
			                 colors="k")
			# if the separatrix isn't shown, manually add a horizontal line for it
			if not np.any(V_max_grid < v_max_contours[2]):
				ax_image.axhline(ω/k, color="k", linewidth=0.7)
		elif wave_frame:
			ax_image.axhline(ω/k, color="k", linewidth=1.0, linestyle="dashed")

		fig.savefig(filename_format.format(i), dpi=150)
		plt.pause(0.05)
	plt.close(fig)


def periodicize(x: NDArray[float], minimum: float, maximum: float) -> NDArray[float]:
	return minimum + (x - minimum) % (maximum - minimum)


def make_gif(base_filename: str, num_frames: int, frame_rate: float):
	# load each frame and put them in a list
	frames = []
	for i in range(num_frames):
		frame = imread(f"{base_filename}_at_t{i:03d}.png")
		rgb = frame[:, :, :3]
		alpha = frame[:, :, 3, newaxis]/255.
		frame = (rgb*alpha + 255*(1 - alpha)).astype(uint8) # remove transparency with a white background
		frames.append(frame)
	# also put a flash of white at the end if the image is transient
	if "wave" in base_filename:
		for i in range(int(frame_rate/10)):
			frames.append(full(shape(frames[0]), 255, dtype=uint8))
	# save it all as a GIF
	mimsave(f"{base_filename}.gif", frames, fps=frame_rate)


def format_as_fraction(coefficient: float, numerator: str, denominator: str):
	if coefficient == 0:
		return "$0$"
	if coefficient < 0:
		return f"$-{format_as_fraction(abs(coefficient), numerator, denominator)[1:]}"
	for base in range(1, 10):
		if coefficient % (1/base) < 1e-10:
			if round(coefficient*base) != 1:
				numerator = f"{round(coefficient*base):d} {numerator}"
			if base != 1:
				denominator = f"{base:d} {denominator}"
			return f"$\\frac{{{numerator}}}{{{denominator}}}$"
	raise ValueError(f"couldn't find a good fraction representation of {coefficient}.")


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="{asctime} | {levelname:4s} | {message}",
		style="{")
	main()
