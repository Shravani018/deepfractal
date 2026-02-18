"""
Fractal 1: Neural Forward Pass Dynamics

For each point in the complex plane, c defines the weight w = c, with bias coupled as b = c * 0.3j

We iterate the neural layer z -> tanh(w*z + b) starting from z = 0.3 and track two things:
1. Escape speed: how fast z grows beyond a threshold (diverges)
2. Orbit trap: how close z gets to the unit circle during iteration

Points that escape quickly are bright. Points that orbit near the unit circle glow from the inside. 
The boundary between the two is fractal, and marks exactly where a neural layer transitions from stable to chaotic.
"""
# Importing necessary libraries
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
# Parameters
img_w, img_h=1200, 1200 # Image width and height
max_iter=200 # Maximum iterations
escape_r=4.0 # Escape radius
bounds=(-2.5, 2.5, -2.5, 2.5) # (xmin, xmax, ymin, ymax)
# Creating an output directory
os.makedirs("./outputs", exist_ok=True)
# Color palette
palette=LinearSegmentedColormap.from_list("", [
    "#000000","#03000f","#0d0030","#2a0060",
    "#6600bb","#cc00ff","#ff44aa","#ff8800",
    "#ffdd00","#fff8cc","#ffffff"
], N=4096)
# Creating the grid
x0, x1, y0, y1=bounds
# Each point c in the complex plane sets w = c and b = c * 0.3j
c=(np.linspace(x0, x1, img_w)[np.newaxis,:] + 1j*np.linspace(y0, y1, img_h)[:,np.newaxis])
w, b=c, c * 0.3j # weights and biases
z=np.full_like(c, 0.3+0.0j) # input
# Initialize arrays for escape time, smooth coloring, orbit trap, final angle, and escape mask
smooth=np.zeros((img_h, img_w), dtype=float)
trap_min=np.full((img_h, img_w), np.inf)
ang_fin=np.zeros((img_h, img_w), dtype=float)
escaped=np.zeros((img_h, img_w), dtype=bool)
# Iterating the function z -> tanh(w*z + b) and applying escape time and orbit trap coloring
for i in range(max_iter):
    mask=~escaped
    if not np.any(mask): break
    z_new=np.tanh(w[mask]*z[mask] + b[mask])
    z[mask]=np.where(np.isfinite(z_new), z_new, 0+0j)
    absz=np.abs(z)
    # Orbit trap: tracking minimum distance to unit circle, only for still-active points
    active=mask & ~escaped
    trap_min=np.where(active & (np.abs(absz-1.0) < trap_min), np.abs(absz-1.0), trap_min)
    new=active & (absz > escape_r)
    if np.any(new):
        lz=np.log(np.maximum(absz[new], 1e-10))
        smooth[new]=np.clip(i+1-(lz-np.log(escape_r))/np.log(escape_r), 0, max_iter)
        ang_fin[new]=np.angle(z[new])
        escaped[new]=True
# Mapping escape time, angle, and orbit trap to color
si=smooth / max_iter
ang=(ang_fin + np.pi) / (2*np.pi)
tm=np.log1p(trap_min) / np.log1p(escape_r)
img=np.where(escaped, si**0.5 + 0.12*np.sin(ang*np.pi*6 + si*12), 0.06*(1-tm)**2)
img=np.clip(img, 0, 1)
glow=gaussian_filter(img, sigma=3)
img=np.clip(img + 0.3*glow, 0, 1)
y_, x_=np.ogrid[:img_h, :img_w]
vig=np.clip(1 - 0.45*((((x_-img_w/2)/(img_w/2))**2+((y_-img_h/2)/(img_h/2))**2)), 0, 1)
img*=vig
# Plotting the fractal
fig, ax=plt.subplots(figsize=(10, 10), facecolor="#000000")
ax.imshow(img, cmap=palette, origin="lower", interpolation="lanczos",
          vmin=0, vmax=1, extent=[x0, x1, y0, y1])
ax.set_title("Forward Pass Dynamics: z -> tanh(w*z + b)",color="white", fontsize=13, fontweight="bold",fontfamily="monospace", pad=14)
ax.set_xlabel("Re(w)", color="#666688", fontsize=9, fontfamily="monospace")
ax.set_ylabel("Im(w)", color="#666688", fontsize=9, fontfamily="monospace")
ax.tick_params(colors="#444466", labelsize=7)
for s in ax.spines.values(): s.set_edgecolor("#222233")
fig.tight_layout()
plt.savefig("./outputs/forward_pass.png",
            dpi=180, bbox_inches="tight", facecolor="black")
print("Fractal image saved to ./outputs/forward_pass.png")