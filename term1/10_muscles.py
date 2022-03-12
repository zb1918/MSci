from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import constants

plt.style.use("cool-style.mplstyle")


fig, ax = plt.subplots()
spec = fits.getdata('term1/muscles/hlsp_muscles_multi_multi_gj436_broadband_v22_adapt-const-res-sed.fits', 1)
plt.plot(spec['WAVELENGTH'], spec['FLUX'] * spec['WAVELENGTH'])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux Density (J/cm2/s)')
ax.ticklabel_format(style = 'plain', axis = 'x')

plt.show()

