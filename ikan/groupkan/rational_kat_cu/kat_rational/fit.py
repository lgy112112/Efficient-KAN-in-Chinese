import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.ticker import FormatStrFormatter

# Define the complex function model
def complex_function(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4):
    numerator = a0 + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4) + a5 * (x**5)
    denominator = 1 + np.abs(b1 * x + b2 * (x**2) + b3 * (x**3) + b4 * (x**4))
    return (numerator / denominator)

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))

def silu(x):  # also known as Swish
    return x / (1 + np.exp(-x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def GEGLU(x):
    return gelu(x) * x

def ReGLU(x):
    return relu(x) * x 

def Swish(x):
    return x / (1 + np.exp(-x))

def SwishGLU(x):
    return Swish(x) * x

def sin(x):
    return np.sin(x*5)

def  erfc_Softplus_2(x):
    # erfc(Softplus(x))2
    return (1 - erf(np.log(1 + np.exp(x))))**2

# Plotting enhancements
def plot_results(x_data, y_data, y_fitted, function_name):
    label_size = 24
    legend_size = 18
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'r-', label=f'{function_name} Function', linewidth=4)
    plt.plot(x_data, y_fitted, 'b--', label='Fitted Rational', linewidth=3)

    # Add details for better readability and presentation
    # plt.title(f'Fitting Complex Function to {function_name}', fontsize=16)
    plt.xlabel('x', fontsize=label_size)
    plt.ylabel('y', fontsize=label_size)
    plt.legend(fontsize=label_size)
    plt.grid(True)  # Turn on grid
    plt.tight_layout()  # Adjust layout to not cut off elements

    # Increase tick font size
    plt.xticks(fontsize=legend_size)
    plt.yticks(fontsize=legend_size)

    # plt.show()
    plt.savefig(f'{function_name}_fit.pdf')

# Plotting enhancements with MSE calculation
def plot_results_with_mse(x_data, y_data, y_fitted, function_name):
    # Calculate MSE
    mse = np.mean((y_data - y_fitted) ** 2)

    label_size = 24
    legend_size = 24
    plt.figure(figsize=(10, 6))

    # Plot original and fitted data
    plt.plot(x_data, y_data, 'r-', label=f'{function_name} Function', linewidth=4)
    plt.plot(x_data, y_fitted, 'b--', label='Fitted Rational', linewidth=3)

    # Annotate plot with MSE in scientific notation at bottom-right corner
    plt.text(0.95, 0.05, f'MSE: {mse:.2e}', fontsize=legend_size, transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='bottom', 
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    # Label axes and add legend
    plt.xlabel('x', fontsize=label_size)
    plt.ylabel('y', fontsize=label_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)  # Turn on grid
    

    # Adjust tick font size
    plt.xticks(fontsize=legend_size)
    plt.yticks(fontsize=legend_size)
    plt.tight_layout()  # Adjust layout to prevent cut-off
    # Format y-axis ticks to 1 decimal place
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Save plot to file
    plt.savefig(f'{function_name}_fit_with_mse.pdf')
    
# Function to fit and plot
def fit_and_plot_activation(function_name):
    # Select the activation function
    activation_functions = {
        'ReLU': relu,
        'GELU': gelu,
        'SiLU': silu,
        'Mish': mish,
        'GEGLU': GEGLU,
        'ReGLU': ReGLU,
        'Swish': Swish,
        'SwishGLU': SwishGLU,
        'sin': sin,
        'erfc_Softplus_2': erfc_Softplus_2
    }
    
    if function_name not in activation_functions:
        print("Invalid function name. Choose 'ReLU', 'GELU', or 'SiLU'.")
        return

    activation_func = activation_functions[function_name]

    # Generate sample data
    x_data = np.linspace(-3, 3, 1000)
    y_data = activation_func(x_data)

    # Initial parameter guesses
    initial_guesses = [-4.41576808e+01 , 2.81414579e+04 , 1.42970453e+04 ,-1.98068326e+04,
 -2.73484568e+03 ,2.64281693e+03,  1.72808118e+05 ,-9.64022283e+04,
 -1.32828660e+04,  1.00431456e+04]

    # Fit the complex function to the activation function data
    try:
        popt, pcov = curve_fit(complex_function, x_data, y_data, p0=initial_guesses)
        print(f"Fitted coefficients for {function_name}: {popt}")
    except Exception as e:
        print(f"An error occurred during fitting: {e}")
        return

    # Generate y values from the fitted model
    y_fitted = complex_function(x_data, *popt)

    # Enhanced plotting function
    plot_results_with_mse(x_data, y_data, y_fitted, function_name)
    
# Example usage
# fit_and_plot_activation('ReLU')
# fit_and_plot_activation('GELU')
# # fit_and_plot_activation('SiLU')
# fit_and_plot_activation('Swish')
# # fit_and_plot_activation('Mish')
# fit_and_plot_activation('GEGLU')
# fit_and_plot_activation('ReGLU')
# fit_and_plot_activation('SwishGLU')
# fit_and_plot_activation('erfc_Softplus_2')
fit_and_plot_activation('sin')

