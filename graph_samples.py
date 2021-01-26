from itertools import product

import matplotlib.pyplot as plt

from experiments import distributions, generate_xs_ys

noise_dists = [
    'uniform(-1,1)',
    'normal(0,.5^2)',
    'beta(2-2)-mirrored',
    'power-law(2)-mirrored'
]

x_distribution = distributions['uniform(-2,2)']

noise_distributions = {name: dist for name, dist in distributions.items() if name in noise_dists}

slopes = [0, 1, 5]


def display_noise_dist(noise_dist, slope, num_points=500):
    xs, ys = generate_xs_ys(num_points, noise_dist, slope, x_distribution)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.scatter(xs, ys)


def save_all_noise_dists():
    for (dist_name, noise_dist), slope in product(noise_distributions.items(), slopes):
        display_noise_dist(noise_dist, slope)
        plt.savefig(f'point_samples/slope-{slope}-{dist_name}.svg', format='svg')


save_all_noise_dists()
