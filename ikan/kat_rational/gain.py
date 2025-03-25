import numpy as np
import json
import os
from scipy.special import erf
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

cfd = os.path.dirname(os.path.realpath(__file__))
with open(f'{cfd}/init.json') as json_file:
    data = json.load(json_file)
    
def rational(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4):
    numerator = a0 + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4) + a5 * (x**5)
    denominator = 1 + np.abs(b1 * x + b2 * (x**2) + b3 * (x**3) + b4 * (x**4))
    return (numerator / denominator)

def gelu(x):
    return F.gelu(torch.tensor(x)).numpy()


def swish(x):
    return x * torch.sigmoid(torch.tensor(x)).numpy()

def calculate_gain(name):
    a = data[name]['init_w_numerator']
    b = data[name]['init_w_denominator']
    
    # Parameters
    n_samples = 1000000
    sigma = 1  # Standard deviation of the input
    x_samples = np.random.normal(0, sigma, n_samples)
    
    rational_values = rational(x_samples, a[0], a[1], a[2], a[3], a[4], a[5], b[0], b[1], b[2], b[3])
    
    # Calculate the gain
    gain = sigma**2 / np.mean(rational_values**2)
    print(f'Gain for {name}: {gain}')
    return gain, x_samples, rational_values

calculate_gain('gelu')
calculate_gain('swish')
calculate_gain('swishglu')
calculate_gain('geglu')
calculate_gain('tanh')
calculate_gain('sigmoid')

# plot

# gain, x_samples, rational_values = calculate_gain('gelu')
# plt.plot(x_samples, rational_values, 'r.', label='Rational Function')
# plt.plot(x_samples, gelu(x_samples), 'b.', label='GELU Function')
# plt.plot(x_samples, swish(x_samples), 'b.', label='Swish Function')

# plt.xlim(-3, 3)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
