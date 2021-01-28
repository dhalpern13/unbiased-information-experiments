import json
import uuid
from itertools import count, product
from os import path
from sys import argv

import pandas as pd
from scipy.stats import uniform, norm, beta, bernoulli, powerlaw

from algorithms import all_points, partition, random_subsets, opt

all_algorithms = {
    'all': all_points,
    'partition': partition,
    'random_1': random_subsets(lambda m, k: 1),
    'random_(10*km)': random_subsets(lambda m, k: 10 * k * m),
    'random_1000': random_subsets(lambda m, k: 1000),
    'opt': opt,
}


def mirror_distribution(dist):
    def sample(shape):
        return dist.rvs(shape) * (bernoulli(.5).rvs(shape) * 2 - 1)

    return sample


distributions = {
    'uniform(-1,1)': uniform(-1, 2).rvs,
    'uniform(-2,2)': uniform(-2, 4).rvs,
    'normal(0,.5^2)': norm(0, .5).rvs,
    'beta(2-2)-mirrored': mirror_distribution(beta(2, 2)),
    'power-law(2)-mirrored': mirror_distribution(powerlaw(3)),
}


def generate_xs_ys(num_points, noise_distribution, slope, x_distribution):
    xs = x_distribution((num_points, 1))
    noise = noise_distribution(num_points)
    ys = xs.dot([slope]) + noise
    return xs, ys


def run_experiment(algorithms, x_distribution, noise_distribution, slope, m, k):
    xs, ys = generate_xs_ys(num_points=m * k,
                            noise_distribution=noise_distribution,
                            slope=slope,
                            x_distribution=x_distribution)
    return {alg_name: alg(xs, ys, m) for alg_name, alg in algorithms.items()}


def compute_row(algorithms, x_distribution, noise_distributions, noise_distribution_name, slope, m, k):
    alg_intercepts = run_experiment(algorithms=algorithms,
                                    x_distribution=x_distribution,
                                    noise_distribution=noise_distributions[noise_distribution_name],
                                    slope=slope,
                                    m=m,
                                    k=k)

    row = {
              'row_id': uuid.uuid4(),
              'noise_distribution': noise_distribution_name,
              'slope': slope,
              'm': m,
              'k': k
          } | alg_intercepts

    return row


def main():
    conf_file = argv[1]

    with open(conf_file) as f:
        conf = json.load(f)

    output_file = path.join('data', conf['output_file'])

    algorithms = {name: alg for name, alg in all_algorithms.items() if name in conf['algorithms']}
    x_distribution = distributions[conf['x_distribution']]
    noise_distributions = {name: dist for name, dist in distributions.items() if name in conf['noise_distributions']}

    k_values = conf['k_values']
    m_values = conf['m_values']
    slopes = conf['slopes']

    for i in count(start=1):
        results = [compute_row(algorithms, x_distribution, noise_distributions, *params) for params in
                   product(noise_distributions.keys(), slopes, m_values, k_values)]

        if path.exists(output_file):
            extra_kwargs = {
                'mode': 'a',
                'header': False
            }
        else:
            extra_kwargs = {}

        pd.DataFrame(results).to_csv(output_file, index=False, **extra_kwargs)

        print(f'{i} iterations complete')


if __name__ == '__main__':
    main()
