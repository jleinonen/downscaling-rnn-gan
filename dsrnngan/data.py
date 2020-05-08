import numpy as np
from scipy.ndimage import convolve


class Smoothener(object):
    def __init__(self):
        (x,y) = np.mgrid[-2:3,-2:3]
        self.smoothing_kernel = np.exp(-0.5*(x**2+y**2)/(0.65**2))
        self.smoothing_kernel /= self.smoothing_kernel.sum()
        self.edge_shapes = {}

    def smoothen(self, img):
        img_shape = tuple(img.shape[2:4])
        if img_shape not in self.edge_shapes:
            s = convolve(np.ones(img_shape, dtype=np.float32),
                self.smoothing_kernel, mode="constant")
            s = 1.0/s
            self.edge_shapes[img_shape] = s
        else:
            s = self.edge_shapes[img_shape]

        img_smooth = np.empty_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[-1]):
                    img_smooth[i,j,:,:,k] = convolve(img[i,j,:,:,k],
                        self.smoothing_kernel, mode="constant") * s

        return img_smooth


class BatchGenerator(object):
    
    def __init__(self, sequences, decoder, downsampler, batch_size=32,
        random_seed=None, augment=True, smoothen_image=True, zeros_frac=0.0):

        self.batch_size = batch_size
        self.sequences = sequences
        self.N = self.sequences.shape[0]
        self.img_shape = tuple(self.sequences.shape[2:4])
        self.num_frames = self.sequences.shape[1]
        self.decoder = decoder
        self.downsampler = downsampler
        self.augment = augment
        self.smoothen_image = smoothen_image
        self.zeros_frac = zeros_frac
        self.smoothener = Smoothener()
        self.reset()

    def __iter__(self):
        return self

    def reset(self, random_seed=None):
        self.prng = np.random.RandomState(seed=random_seed)
        self.next_ind = np.array([], dtype=int)

    def next_indices(self):
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(self.N, dtype=int)
            self.prng.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])
        return self.next_ind[:self.batch_size]

    def __next__(self):
        ind = self.next_indices()
        self.next_ind = self.next_ind[self.batch_size:]

        X = self.sequences[ind,...]

        X = self.decoder(X)
        if self.augment:
            X = self.augment_sequence_batch(X)
        Y = self.downsampler(X)
        X = self.decoder.normalize(X)
        Y = self.decoder.normalize(Y)
        if self.smoothen_image:
            X = self.smoothener.smoothen(X)

        if self.zeros_frac > 0.0:
            set_zero = (self.prng.rand(X.shape[0]) < self.zeros_frac)
            X[set_zero,...] = 0.0
            Y[set_zero,...] = 0.0

        return (X,Y)

    def augment_sequence(self, sequence):
        seq = sequence.copy()

        # mirror
        if bool(self.prng.randint(2)):
            seq = np.flip(seq, axis=1)
        if bool(self.prng.randint(2)):
            seq = np.flip(seq, axis=2)

        # rotate
        num_rot = self.prng.randint(4)
        if num_rot > 0:
            seq = np.rot90(seq, k=num_rot, axes=(1,2))

        return seq

    def augment_sequence_batch(self, sequences):
        sequences = sequences.copy()
        for i in range(sequences.shape[0]):
            sequences[i,...] = self.augment_sequence(sequences[i,...])
        return sequences


class RainRateDecoder(object):
    def __init__(self, scaling_fn, value_range=(np.log10(0.1), np.log10(100)),
        below_val=np.nan, normalize=False):

        self.logR = np.log10(np.load(scaling_fn))
        self.logR[0] = np.nan
        #self.x = np.arange(len(self.logR))
        self.value_range = value_range
        self.below_val = below_val
        self.normalize_output = normalize

    def __call__(self, img):
        valid = (img != 0)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = self.logR[img[valid]]
        img_dec[img_dec<self.value_range[0]] = self.below_val
        img_dec.clip(max=self.value_range[1], out=img_dec)
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img-self.below_val) / \
            (self.value_range[1]-self.below_val) 

    def denormalize(self, img, set_nan=True):
        img = img*(self.value_range[1]-self.below_val) + self.below_val
        img[img < self.value_range[0]] = self.below_val
        if set_nan:
            img[img == self.below_val] = np.nan
        return img


class CODDecoder(RainRateDecoder):
    def __init__(self,
        value_range=(np.log(1.19), np.log(158.48865)),
        below_val=np.nan, normalize=False,
        scale_factor=158.48865/(2**16-2)):

        self.value_range = value_range
        self.below_val = below_val
        self.normalize_output = normalize
        self.scale_factor = scale_factor

    def __call__(self, img):
        valid = (img != 0)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = np.log(img[valid]*self.scale_factor)
        img_dec[(img_dec<self.value_range[0]) | ~valid] = self.below_val
        img_dec.clip(max=self.value_range[1], out=img_dec)
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec


class LogDownsampler(object):
    def __init__(self, pool_size=16, min_val=np.nan, threshold_val=None):
        self.pool_size = pool_size 
        self.min_val = min_val
        self.threshold_val = threshold_val

    def __call__(self, log_R):
        R = 10**log_R
        R[~np.isfinite(R)] = 0.0
        lores_shape = (log_R.shape[0], log_R.shape[1], 
            log_R.shape[2]//self.pool_size, log_R.shape[3]//self.pool_size,
            log_R.shape[4])
        R_ds = np.zeros(lores_shape, dtype=np.float32)
        for (il,ih) in enumerate(range(0,log_R.shape[2],self.pool_size)):
            for (jl,jh) in enumerate(range(0,log_R.shape[3],self.pool_size)):
                R_ds[:,:,il,jl,:] = R[:,:,ih:ih+self.pool_size,
                    jh:jh+self.pool_size,:].mean(axis=(2,3))
        log_R_ds = np.log10(R_ds)
        min_mask = ~np.isfinite(log_R_ds)
        if self.threshold_val is not None:
            min_mask |= (log_R_ds < self.threshold_val)
        log_R_ds[min_mask] = self.min_val
        return log_R_ds
