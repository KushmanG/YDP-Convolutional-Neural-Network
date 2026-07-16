'''
In ML you want your input pipeline to be lossless and under your control up to the last moment. Every 
transform should be a deliberate choice you can change, not something a plotting library decided.

That's why we ain't using snapshot.png directly, instead we are converting positions.npy into an image, a 
consumable for the CNN via rasterization

This file's aim is to take a positions.npy file that contains an (N,2) array that holds the position of each
particle and rasterize it.
'''
from a_imports import *

# To compute the box length, using Weinger-Seitz radius
def length(N):
    return np.sqrt(N*np.pi)

# Loading data
def load_sample(sample_dir):
    positions = np.load(os.path.join(sample_dir, "positions.npy"))
    metadata = np.load(os.path.join(sample_dir, "metadata.npy"))

    gamma = float(metadata[0])
    kappa = float(metadata[1])
    N = int(metadata[2])

    return positions, gamma, kappa, N

# Rasterization ***
'''
Resolution is an arbitrary choice

You can change the blob_sigma value just don't put it as None (obviously) or lesser than 0, that's js invalid

===== BLOB_SIGMA TUNING GUIDE (Gaussian smoothing width, in PIXELS)
 Ruler: at 128px, neighbours sit ~8px apart. Keep sigma well under that
 or blobs merge and the lattice melts. Safe window ~ (0, 3]; default 1.0.

 INCREASE sigma when:
   - resolution drops (bigger pixels -> more aliasing to smooth over)
   - overfitting: sharp dots let the CNN memorize a seed's exact fingerprint;
     more blur keeps only the density field -> learns physics, not the frame
   - you want different seeds of the same (G,k) to look more alike
   - model is drowning in sparse-dot noise (cheap fix before touching arch)

 DECREASE sigma (toward 0) when:
   - crystals vs liquids are being confused: high-G signal IS the sharp
     triangular lattice; too much blur erases it (same failure as over-normalizing)

 It's a noise-robustness (big) vs structural-sharpness (small) dial.
 Validate by eye on the gut-check image, then tune on val error.

======== NORMALIZE FLAG TUNING GUIDE (bool: rescale mean pixel -> 1) ---
 NOT a dial, just on/off. Rule: TRUE when the image feeds the model,
 FALSE when it feeds your eyes or a test.

 TRUE (default) when:
   - feeding the CNN: nets train best on O(1) inputs; raw mean is ~0.016
   - it's a FIXED rescale (mass is pinned to N), identical for every image,
     so it fixes scale WITHOUT touching the crystal-vs-liquid signal

 FALSE when:
   - debugging resolution: raw counts show collisions (max>1 = particles
     sharing a pixel = resolution too coarse)
   - testing mass conservation: img.sum() should equal N (=256) exactly
   - you plan to normalize globally in the dataset layer instead

 DO NOT swap this for per-image (img - mean)/std: liquid vs crystal differ
 BY their variance -> std-normalizing erases the Γ signal you're predicting.
'''
def rasterize(positions, N = None, resolution = 128, blob_sigma = 1.0, normalize = True):
    # If N value is not given then it will compute it 
    if N is None:
        N = len(positions)

    # And thus compute the box length
    l = length(N)

    '''
    So to rasterize, we plot the positions in a histogram, it divides the box into a 128x128 (or wtv is the reolution)
    grid and checks if there is any particle in that pixel or nah. This gives a H[x,y] histogra that acts as an image
    corresponding to positions data that is used in the CNN

    Now the np.histogram2d() function returns H[x,y] but obviously for images we need it in [y,x] format so we take the 
    transpose of that histogram.
    '''
    H, _, _ = np.histogram2d(positions[:, 0], positions[:,1], bins = resolution, range = [[0.0, l], [0.0, l]])
    img = H.T

    # this is a general transform cz the histogram gives a density image, pixels with different intensities of colour
    # so the gaussian filter turns the density image into particles in a box image
    img = gaussian_filter(img, sigma = blob_sigma, mode = "wrap")


    # Normalization
    if normalize:
        total = float(img.sum())
        if total > 0.0: #If the img is not empty
            img = img * (img.size / total)
    
    return img.astype(np.float32)

# Credits to Claude Opus 4.8 for rasterized output formatting :D
# It tried its best to explain, imma move on with other work
if __name__ == "__main__":
    # here = the folder this script lives in (CNN/), so paths work from any cwd.
    # ".." climbs out of CNN/ before descending into the sim's dataset.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Simulator", "sample_dataset")
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Which samples to eyeball. 0 = G2 (liquid), 9 = G49, 18 = G1200 (crystal), all k=1.
    # This is the main knob: swap in any sample ids you want to inspect.
    picks = [0, 9, 18]

    # One row per sample, 3 columns: snapshot.png | hard histogram | soft splat.
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))

    for row, sid in enumerate(picks):
        sdir = os.path.join(data_dir, f"sample_{sid:04d}")
        positions, gamma, kappa, N = load_sample(sdir)

        hard = rasterize(positions, N=N, blob_sigma=0.0)   # raw dots, no blur
        soft = rasterize(positions, N=N, blob_sigma=1.0)   # blurred
        snap = plt.imread(os.path.join(sdir, "snapshot.png"))

        # origin="lower" ONLY on our rasters: undoes the array-is-y-down convention
        # so up-on-screen = up-in-physics. The snapshot is already a finished image.
        axes[row, 0].imshow(snap)
        axes[row, 1].imshow(hard, origin="lower", cmap="magma")
        axes[row, 2].imshow(soft, origin="lower", cmap="magma")

        # Label each row with its physics on the left edge.
        axes[row, 0].set_ylabel(f"G={gamma:g}  k={kappa:g}", fontsize=11)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Column headers, top row only.
    axes[0, 0].set_title("snapshot.png")
    axes[0, 1].set_title("hard histogram")
    axes[0, 2].set_title("soft splat (sigma=1)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "rasterize_check.png")
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")

