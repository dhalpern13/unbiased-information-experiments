from itertools import product

import matplotlib.pyplot as plt

from experiments import noise_distributions, x_distribution, generate_xs_ys, slopes


def display_noise_dist(noise_dist, slope, num_points=500):
    xs, ys = generate_xs_ys(num_points, noise_dist, slope, x_distribution)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.scatter(xs, ys)


def save_all_noise_dists():
    for (dist_name, noise_dist), slope in product(noise_distributions.items(), slopes):
        display_noise_dist(noise_dist, slope)
        plt.savefig(f'point_samples/slope-{slope}-{dist_name}.pdf', format='pdf')


save_all_noise_dists()
