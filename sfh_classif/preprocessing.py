import numpy as np


def load_sfh_data(filename):
    """Load SFH data from an individual file
    
    Parameters
    ----------
    filename : str
        Path to SFH data file.
    
    Returns
    -------
    time : array_like
    mass : array_like
        The time and mass arrays from the SFH data file.
    """
    time, mass, _ = np.loadtxt(filename, unpack=True)
    return time, mass


def create_regular_mass_grid(time, mass, max_time):
    """Create a mass table with a regular time step of 1 My
    
    Parameters
    ----------
    time : array_like
        The time array from the SFH data.
    mass : array_like
        The mass array from the SFH data.        
    max_time : int
        The maximum time used to determine the number of time steps.

    Returns
    -------
    grid_mass : array_like
        The mass array with regular 1 My time steps.

    """
    grid_mass = np.zeros(max_time)
    indices = time.astype(int)
    grid_mass[indices] = mass
    return grid_mass


def resample_in_time(grid_mass, n_years=10):
    """Resample the mass table by a number of years
    
    Parameters
    ----------
    grid_mass : array_like
        The mass array with regular 1 My time steps.
    n_years : int, optional
        The number of years to bin the mass table, by default 10

    Returns
    -------
    binned_time : array_like
        The time array binned by n_years.
    binned_mass : array_like
        The mass array binned by n_years.
        
    """
    size = grid_mass.size
    new_size = size // n_years
    binned_mass = np.empty(new_size)
    binned_idx = np.repeat(np.arange(new_size), n_years)
    for i in range(new_size):
        idx = binned_idx == i
        binned_mass[i] = np.sum(grid_mass, where=idx)

    binned_time = np.arange(new_size)

    return binned_time, binned_mass
