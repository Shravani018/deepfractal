"""
Fractal 3: Backpropagation Error Flow

For each point in the complex plane, c encodes a (weight, input) pair.
We iterate the backward pass: delta -> tanh'(w*delta) * delta + c
where tanh'(z) = 1 - tanh^2(z) is the derivative of tanh, and delta is the error signal.

This models how an error signal transforms as it flows backward through layers.
Color encodes whether the error explodes, vanishes, or orbits, the same instability
that makes training deep networks hard without careful initialization.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
# Parameters
img_w, img_h=1200, 1200
max_iter=250
escape_r=20.0
bounds=(-2.5, 2.5, -2.5, 2.5)
# Creating output directory
os.makedirs("./outputs", exist_ok=True)
# Color palette
palette=LinearSegmentedColormap.from_list("", [
    "#000000","#000510","#001830","#003355",
    "#005588","#0088bb","#00bbdd","#44eeff",
    "#aaffee","#eeffff","#ffffff"
], N=4096)
# Creating the grid
x0, x1, y0, y1=bounds
c=(np.linspace(x0, x1, img_w)[np.newaxis,:] +
   1j*np.linspace(y0, y1, img_h)[:,np.newaxis])
# w is the weight, encoded in c
w=c
delta=np.full_like(c, 0.1+0.0j) # small initial error signal
# Initialize arrays
smooth=np.zeros((img_h, img_w), dtype=float)
trap_min=np.full((img_h, img_w), np.inf)
ang_fin=np.zeros((img_h, img_w), dtype=float)
escaped=np.zeros((img_h, img_w), dtype=bool)
# Iterating the backward pass: delta -> tanh'(w * delta) * delta + c
# tanh'(z) = 1 - tanh^2(z), the local gradient that scales the error signal
for i in range(max_iter):
    mask=~escaped
    if not np.any(mask): break
    tanh_val=np.tanh(w[mask] * delta[mask])
    dtanh=1.0 - tanh_val**2 # derivative of tanh: how much gradient passes through
    delta[mask]=dtanh * delta[mask] + c[mask]
    delta=np.where(np.isfinite(delta), delta, 0+0j)
    abs_delta=np.abs(delta)
    # Orbit trap: min distance to unit circle tracks where error signal stabilizes
    trap_min=np.where(mask & (np.abs(abs_delta-1.0) < trap_min), np.abs(abs_delta-1.0), trap_min)
    new=mask & (abs_delta > escape_r)
    if np.any(new):
        lz=np.log(np.maximum(abs_delta[new], 1e-10))
        smooth[new]=np.clip(i+1-np.log(lz/np.log(escape_r))/np.log(2), 0, max_iter)
        ang_fin[new]=np.angle(delta[new])
        escaped[new]=True
# Mapping escape time, angle, and orbit trap to color
si=smooth / max_iter
ang=(ang_fin + np.pi) / (2*np.pi)
tm=np.log1p(trap_min) / np.log1p(escape_r)
img=np.where(escaped, si**0.5 + 0.15*np.sin(ang*np.pi*7 + si*10), 0.07*(1-tm)**2)
img=np.clip(img, 0, 1)
glow=gaussian_filter(img, sigma=3)
img=np.clip(img + 0.3*glow, 0, 1)
y_, x_=np.ogrid[:img_h, :img_w]
vig=np.clip(1 - 0.45*((((x_-img_w/2)/(img_w/2))**2+((y_-img_h/2)/(img_h/2))**2)), 0, 1)
img *= vig
# Plotting the fractal
fig, ax=plt.subplots(figsize=(10, 10), facecolor="#000000")
ax.imshow(img, cmap=palette, origin="lower", interpolation="lanczos",
          vmin=0, vmax=1, extent=[x0, x1, y0, y1])
ax.set_title("Backpropagation Error Flow: delta -> tanh'(w*delta)*delta + c",
             color="white", fontsize=13, fontweight="bold",
             fontfamily="monospace", pad=14)
ax.set_xlabel("Re(w)", color="#224466", fontsize=9, fontfamily="monospace")
ax.set_ylabel("Im(w)", color="#224466", fontsize=9, fontfamily="monospace")
ax.tick_params(colors="#223344", labelsize=7)
for s in ax.spines.values(): s.set_edgecolor("#112233")
fig.tight_layout()
plt.savefig("./outputs/backprop.png",
            dpi=180, bbox_inches="tight", facecolor="black")
print("Fractal image saved to ./outputs/backprop.png")