import numpy as np


def generate_divisors(n):
    # adapted from: https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number
    for i in range(1, int(n/2) + 1):
        if n % i == 0:
            yield i
    yield n


def std_error(numbers):
    numbers = np.array(numbers)
    mu = numbers.mean()
    std = (numbers.var())**0.5
    std_error = std/(len(numbers)**0.5)
    return f"{mu:.2e}{chr(177)}{std_error:.2e}", mu, std_error


def get_geometric_decay(n_steps, initial_value, final_value):
    # initial_value * decay ^ {n_steps} = final_value
    return (final_value / initial_value) ** (1./n_steps)
