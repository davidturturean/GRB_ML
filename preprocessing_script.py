# WITH THIS SCRIPT I AM CONSTRUCTING THE DATASETS FOR TRAINING

#----------------------------------------------
#IMPORTING
from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
import os

dr12q_path = '/Users/research/Documents/DR12Q Training Dupe/data/raw/DR12Q.fits'
processed_dir = '/Users/research/Documents/DR12Q Training Dupe/data/processed'

# if os.path.exists(dr12q_path):
#     hdulist = fits.open(dr12q_path)
#     data = hdulist[1].data
# else:
#     print(f"File not found: {dr12q_path}")
#     exit(1)

# if not os.path.exists(processed_dir):
#     os.makedirs(processed_dir)

hdulist = fits.open(dr12q_path)
data = hdulist[1].data

fluxes = data['PSFFLUX']
#fluxes are in u,g,r,i,z bands

specific_wavelengths = np.array([354.3, 477.0, 623.1, 762.5, 913.4])
wavelength_start = specific_wavelengths[0]
wavelength_end = specific_wavelengths[-1]
num_data_points = 4618

redshifts = data['Z_VI']
#can also do 'Z_PIPE' for BOSS pipeline redshift, 'Z_PCA' PCA redshift


#-------------------------------------------------
#ACTUAL PROCESSING

log_wavelengths = np.linspace(np.log10(wavelength_start), np.log10(wavelength_end), num_data_points)
uniform_wavelengths = 10**log_wavelengths

def interpolate_and_normalize(flux):
    # Interpolating flux values to uniform_wavelengths
    interpolated_flux = interp1d(specific_wavelengths, flux, bounds_error=False, fill_value="extrapolate")(uniform_wavelengths)
    
    mean_flux = np.mean(interpolated_flux)
    std_flux = np.std(interpolated_flux)
    normalized_flux = (interpolated_flux - mean_flux) / std_flux if std_flux != 0 else interpolated_flux - mean_flux
    return normalized_flux

normalized_fluxes = []
valid_redshifts = []

for flux, redshift in zip(fluxes, redshifts):
    normalized_flux = interpolate_and_normalize(flux)
    normalized_fluxes.append(normalized_flux)
    valid_redshifts.append(redshift)
    
normalized_fluxes = np.array(normalized_fluxes)
valid_redshifts = np.array(valid_redshifts)

# Ensuring the dimensions match (number of samples, data points)
assert normalized_fluxes.shape[1] == num_data_points, "Each spectrum should have 4618 data points"
        
#--------------------------------------------
#EXPORTING TRAINING, VALIDATING (AND TESTING) DATASETS

train_size = int(0.9 * len(normalized_fluxes))
val_size = int(0.1 * len(normalized_fluxes))
#test_size = len(normalized_fluxes) - train_size - val_size

train_fluxes = normalized_fluxes[:train_size]
val_fluxes = normalized_fluxes[train_size:train_size+val_size]
#test_fluxes = normalized_fluxes[(train_size+val_size):]

train_labels = redshifts[:train_size]
val_labels = redshifts[train_size:train_size+val_size]
#test_labels = redshifts[train_size+val_size:]

processed_dir = '/Users/research/Documents/DR12Q Training Dupe/data/processed'

#print(np.shape(train_fluxes), np.shape(val_fluxes), np.shape(test_fluxes))

# Save the preprocessed data
try:
    np.save(os.path.join(processed_dir, 'train_fluxes.npy'), train_fluxes)
    np.save(os.path.join(processed_dir, 'val_fluxes.npy'), val_fluxes)
    #np.save(os.path.join(processed_dir, 'test_fluxes.npy'), test_fluxes)
    np.save(os.path.join(processed_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(processed_dir, 'val_labels.npy'), val_labels)
    #np.save(os.path.join(processed_dir, 'test_labels.npy'), test_labels)
except Exception as e:
    print(f"An error occurred: {e}")

