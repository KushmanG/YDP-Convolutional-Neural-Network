from a_imports import *
from b_rasterize import rasterize, load_sample

# First thing this needs to do is to read manifest.csv
'''
This function reads the manifest.csv and appends data from it into an array
We read it using the standard file reading method, using the csv.DictReader(file_name) lib

'''
def read_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline = "") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({"sample_id": int(row["sample_id"]), "gamma": float(row["gamma"]), "kappa": float(row["kappa"]), "seed": int(row["seed"])})

    return rows

'''
Now we split the dataset into 3 parts:
1. Train: Model elarns its weights frm this
2. Validation (val): To validate data and fix variables for better performance
3. Test: To test the performance of it

For that we basically have, say, x gamma values, y kappa values and z seeds per gamma-kappa pair
So we choose one gamma kappa pair and split the SEEDS between each split

So we need to iterate through all gamma-kappa pairs and give like 70% to Train, 15% to val, 15% to test
'''

def split(rows, train_fraction = 0.7, val_fraction = 0.15):
    # Stratified split: bucket the rows by their (gamma, kappa) cell first, so the
    # fractions apply WITHIN every cell instead of across the manifest's ordering.
    # This way every cell shows up in every split, no matter how the file is sorted.
    cells = {}
    for row in rows:
        cells.setdefault((row["gamma"], row["kappa"]), []).append(row)

    train, val, test = [], [], []
    for bucket in cells.values():
        n = len(bucket)
        n_train = round(train_fraction * n)
        n_val = round(val_fraction * n)

        # test takes whatever remains (never rounded separately), so the three
        # pieces always add back up to the full bucket
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    return train, val, test


'''
Target encoding/decoding.

Gamma runs from 2 to 1200 — three orders of magnitude. If we regress on raw gamma,
the loss on a G=1200 sample dwarfs the loss on a G=2 sample and the model only
learns crystals. So the model works in "model space": [log10(gamma), kappa].
encode() goes physical -> model space (used here when building targets),
decode() goes model space -> physical (used by eval.py to report real units).
'''
def encode(gamma, kappa):
    return [np.log10(gamma), kappa]

def decode(label):
    return float(10.0 ** label[0]), float(label[1])


class dataset(torch.utils.data.Dataset):
    '''
    The Dataset's whole job: given an index i, hand back ONE (image, target) pair.
    The DataLoader handles batching/shuffling on top of this.

    __init__ just stores the settings — nothing is loaded up front. Each sample is
    loaded and rasterized lazily in __getitem__, so memory stays flat no matter
    how big the dataset gets.
    '''
    def __init__(self, rows, data_dir, resolution = 128, blob_sigma = 1.0, augment = False):
        self.rows = rows
        self.data_dir = data_dir
        self.resolution = resolution
        self.blob_sigma = blob_sigma
        self.augment = augment

    # DataLoader asks this to know how many samples an epoch has
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]

        # Folders are named sample_0000, sample_0001, ... so zero-pad the id to 4 digits
        sample_dir = os.path.join(self.data_dir, f"sample_{row['sample_id']:04d}")
        positions, gamma, kappa, N = load_sample(sample_dir)

        # positions.npy -> (resolution x resolution) float32 image
        img = rasterize(positions, N = N, resolution = self.resolution,
                        blob_sigma = self.blob_sigma, normalize = True)

        '''
        Augmentation (train split only): apply a random symmetry of the periodic box.
        Rotations by 90, flips, and periodic translations (np.roll wraps pixels around,
        matching the wrap-around physics) all change the IMAGE without changing the
        PHYSICS — so the model sees "new" samples that must map to the same target.
        '''
        if self.augment:
            img = np.rot90(img, np.random.randint(4))
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
            img = np.roll(img, shift = (np.random.randint(self.resolution),
                                        np.random.randint(self.resolution)), axis = (0, 1))

        # rot90/fliplr return negative-stride views that torch.from_numpy refuses,
        # ascontiguousarray gives a clean copy (and is a no-op if already clean)
        img = np.ascontiguousarray(img)

        # unsqueeze(0) adds the channel dim: (res, res) -> (1, res, res), since conv1 expects 1 channel
        image = torch.from_numpy(img).unsqueeze(0)
        target = torch.tensor(encode(gamma, kappa), dtype = torch.float32)

        return image, target

# eval.py imports the class under this name — same thing, two handles
PlasmaDataset = dataset