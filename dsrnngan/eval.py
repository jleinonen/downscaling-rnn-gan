from bisect import bisect_left
from datetime import datetime, timedelta
import os

import netCDF4
import numpy as np
from scipy.interpolate import interp1d

import crps
import train
import data
import models
import msssim
import noise
import plots
import rainfarm


path = os.path.dirname(os.path.abspath(__file__))


def randomize_nans(x, rnd_mean, rnd_range):
    nan_mask = np.isnan(x)
    nan_shape = x[nan_mask].shape
    x[nan_mask] = rnd_mean + \
        (np.random.rand(*nan_shape)-0.5)*rnd_range


def ensemble_ranks(gen, batch_gen, noise_gen, 
    noise_offset=0.0, noise_mul=1.0,
    num_batches=1024, rank_samples=100, normalize_ranks=True):

    rnd_range = 0.1 * (batch_gen.decoder.value_range[0] -
        batch_gen.decoder.below_val)

    ranks = []
    crps_scores = []

    for k in range(num_batches):
        (sample,cond) = next(batch_gen)
        sample_crps = sample
        sample = sample.ravel()
        sample = batch_gen.decoder.denormalize(sample)
        randomize_nans(sample, batch_gen.decoder.below_val, rnd_range)

        samples_gen = []
        crps_scores = []
        for i in range(rank_samples):
            n = noise_gen()
            for nn in n:
                nn *= noise_mul
                nn -= noise_offset
            sample_gen = gen.predict([cond]+n)
            samples_gen.append(sample_gen)

        samples_gen = np.stack(samples_gen, axis=-1)

        crps_score = crps.crps_ensemble(sample_crps, samples_gen)
        crps_scores.append(crps_score.ravel())

        samples_gen = samples_gen.reshape(
            (np.prod(samples_gen.shape[:-1]), samples_gen.shape[-1]))
        samples_gen = batch_gen.decoder.denormalize(samples_gen)
        randomize_nans(samples_gen, batch_gen.decoder.below_val, rnd_range)

        rank = np.count_nonzero(sample[:,None] >= samples_gen, axis=-1)
        ranks.append(rank)

    ranks = np.concatenate(ranks)
    crps_scores = np.concatenate(crps_scores)
    
    if normalize_ranks:
        ranks = ranks / rank_samples

    return (ranks, crps_scores)


def rank_KS(norm_ranks, num_ranks=100):
    (h,b) = np.histogram(norm_ranks, num_ranks+1)
    h = h / h.sum()
    ch = np.cumsum(h)
    cb = b[1:]
    return abs(ch-cb).max()


def rank_CvM(norm_ranks, num_ranks=100):
    (h,b) = np.histogram(norm_ranks, num_ranks+1)
    h = h / h.sum()
    ch = np.cumsum(h)
    cb = b[1:]
    db = np.diff(b)
    
    return np.sqrt(((ch-cb)**2*db).sum())


def rank_DKL(norm_ranks, num_ranks=100):
    (h,b) = np.histogram(norm_ranks, num_ranks+1)
    q = h / h.sum()
    p = 1/len(h)
    return p*np.log(p/q).sum()
    

def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks==0) | (norm_ranks==1)
    )
    op = float(op)/len(norm_ranks)
    return op


def rank_metrics_by_time(application, data_file, out_fn,
    weights_dir, check_every=1, N_range=None):
    (wgan, batch_gen_train, batch_gen_valid, batch_gen_test,
        noise_shapes, steps_per_epoch) = train.setup_gan(data_file,
        application=application, batch_size=64)
    gen = wgan.gen
    noise_gen = noise.NoiseGenerator(noise_shapes(),
        batch_size=batch_gen_valid.batch_size)

    files = os.listdir(weights_dir)
    def get_id(fn):
        return fn.split("-")[1]
    files = sorted(fn for fn in files if get_id(fn)==application)

    def log_line(line):
        with open(out_fn, 'a') as f:
            print(line, file=f)
    log_line("N KS CvM DKL OP CRPS mean std")

    for fn in files[::check_every]:
        N_samples = int(fn.split("-")[-1].split(".")[0])
        if (N_range is not None) and not (N_range[0] <= N_samples < N_range[1]):
            continue
        gen.load_weights(weights_dir+"/"+fn)
        (ranks, crps_scores) = ensemble_ranks(gen, batch_gen_valid,
            noise_gen, num_batches=8)
        
        KS = rank_KS(ranks)
        CvM = rank_CvM(ranks) 
        DKL = rank_DKL(ranks)
        OP = rank_OP(ranks)
        CRPS = crps_scores.mean() 
        mean = ranks.mean()
        std = ranks.std()

        log_line("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            N_samples, KS, CvM, DKL, OP, CRPS, mean, std))


def rank_metrics_by_noise(application, run_id, data_file,
    weights_fn):
    (wgan, batch_gen_train, batch_gen_valid, _,
        noise_shapes, steps_per_epoch) = train.setup_gan(data_file,
        application=application)
    gen = wgan.gen
    noise_gen = noise.NoiseGenerator(noise_shapes(),
        batch_size=batch_gen_valid.batch_size)

    for m in list(range(0.5,2.51,0.1))+[3.0,3.5]:
        N_samples = int(fn.split("-")[-1].split(".")[0])
        gen.load_weights(weights_dir+"/"+fn)
        (ranks, crps_scores) = ensemble_ranks(gen, batch_gen_valid,
            noise_gen, num_batches=32, noise_mul=m)
        
        KS = rank_KS(ranks)
        CvM = rank_CvM(ranks) 
        DKL = rank_DKL(ranks)
        CRPS = crps_scores.mean()
        mean = ranks.mean()
        std = ranks.std()

        print(N_samples, KS, CvM, DKL, CRPS, mean, std)


def rank_metrics_table(application, data_file, weights_fn, method="gan"):

    if method=="gan":
        (wgan, batch_gen_train, batch_gen_valid, batch_gen_test,
            noise_shapes, steps_per_epoch) = train.setup_gan(data_file,
            test_data_file=data_file, application=application, batch_size=64)
        gen = wgan.gen
        gen.load_weights(weights_fn)
    elif method=="rainfarm":
        (gen_det, batch_gen_train, batch_gen_valid, 
            batch_gen_test, steps_per_epoch) = train.setup_deterministic(data_file,
            test_data_file=data_file, sample_random=True, n_samples=1, batch_size=64,
            application=application, loss='mse')
        gen = GeneratorRainFARM(16, batch_gen_test.decoder)
        noise_shapes = lambda: []

    noise_gen = noise.NoiseGenerator(noise_shapes(),
        batch_size=batch_gen_valid.batch_size)

    (ranks, crps_scores) = ensemble_ranks(gen, batch_gen_test,
        noise_gen, num_batches=16)
    
    KS = rank_KS(ranks)
    CvM = rank_CvM(ranks) 
    DKL = rank_DKL(ranks)
    OP = rank_OP(ranks)
    CRPS = crps_scores.mean() 
    mean = ranks.mean()
    std = ranks.std()

    print("KS: {:.3f}".format(KS))
    print("CvM: {:.3f}".format(CvM))
    print("DKL: {:.3f}".format(DKL))
    print("OP: {:.3f}".format(OP))
    print("CRPS: {:.3f}".format(CRPS))
    print("mean: {:.3f}".format(mean))
    print("std: {:.3f}".format(std))


def reconstruct_time_series_partial(images_fn, gen, noise_shapes,
    init_model, out_fn,
    time_range, h=None, last_t=None, application="mchrzc", ds_factor=16, n_ensemble=4,
    scaling_fn=path+"/../data/scale_rzc.npy", relax_lam=0.0):

    if application == "mchrzc":
        dec = data.RainRateDecoder(scaling_fn, below_val=np.log10(0.025))
    else:
        raise ValueError("Unknown application.")
    downsampler = data.LogDownsampler(min_val=dec.below_val,
        threshold_val=dec.value_range[0])

    with netCDF4.Dataset(images_fn) as ds_img:
        time = np.array(ds_img["time"][:], copy=False)
        time_dt = [datetime(1970,1,1)+timedelta(seconds=t) for t in time]
        t0 = bisect_left(time_dt, time_range[0])
        t1 = bisect_left(time_dt, time_range[1])
        images = np.array(ds_img["images"][t0:t1,...], copy=False)
        time = time[t0:t1]

    img_shape = images.shape[1:3]
    img_shape = (
        img_shape[0] - img_shape[0]%ds_factor,
        img_shape[1] - img_shape[1]%ds_factor,
    )
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape),
        batch_size=n_ensemble)

    images_ds = np.zeros(
        (images.shape[0],img_shape[0]//ds_factor,img_shape[1]//ds_factor,1),
        dtype=np.uint8
    )
    images_gen = np.zeros(
        (images.shape[0],)+img_shape+(1,n_ensemble),
        dtype=np.uint8
    )

    # this finds the nearest index in the R encoding
    def encoder():
        lR = dec.logR
        ind = np.arange(len(lR))
        ip = interp1d(lR,ind)
        def f(x):
            y = np.zeros(x.shape, dtype=np.uint8)
            valid = (x >= dec.value_range[0])
            y[valid] = ip(x[valid]).round().astype(np.uint8)
            return y
        return f
    encode = encoder()

    for k in range(images.shape[0]):
        print("{}/{}".format(k+1,images.shape[0]))
        img_real = images[k:k+1,:img_shape[0],:img_shape[1],:]
        img_real = dec(img_real)
        img_real = img_real.reshape(
            (1,1)+img_real.shape[1:])
        img_real[np.isnan(img_real)] = dec.below_val
        img_ds = downsampler(img_real)
        img_ds = dec.normalize(img_ds)
        img_ds_denorm = dec.denormalize(img_ds)
        img_ds = np.tile(img_ds, (n_ensemble,1,1,1,1))

        (n_init, n_update) = noise_gen()
            
        if (h is None) or (time[k]-last_t != 600):
            h = init_model.predict([img_ds[:,0,...], n_init])
            
        (img_gen,h) = gen.predict([img_ds, h, n_update])
        if relax_lam > 0.0:
            # nudge h towards null
            h_null = init_model.predict([
                np.zeros_like(img_ds[:,0,...]), n_init
            ])
            h = h_null + (1.0-relax_lam)*(h-h_null)
        img_gen = dec.denormalize(img_gen)
        img_gen = img_gen.transpose((1,2,3,4,0))

        images_ds[k,...] = encode(img_ds_denorm[0,...])
        images_gen[k,...] = encode(img_gen[0,...])
        last_t = time[k]

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_height = ds.createDimension("dim_height", img_shape[0])
        dim_width = ds.createDimension("dim_width", img_shape[1])
        dim_height_ds = ds.createDimension("dim_height_ds",
            img_shape[0]/ds_factor)
        dim_width_ds = ds.createDimension("dim_width_ds",
            img_shape[1]/ds_factor)
        dim_samples = ds.createDimension("dim_samples", images.shape[0])
        dim_ensemble = ds.createDimension("dim_ensemble", n_ensemble)
        dim_channels = ds.createDimension("dim_channels", 1)

        var_params = {"zlib": True, "complevel": 9}

        def create_var(name, dims, **params):
            dtype = params.pop("dtype", np.float32)
            var = ds.createVariable(name, dtype, dims, **params)
            return var

        var_img = create_var("images",
            ("dim_samples","dim_height","dim_width","dim_channels",
                "dim_ensemble"),
            chunksizes=(1,64,64,1,1), dtype=np.uint8, **var_params)
        var_img.units = "Encoded R"
        var_img_ds = create_var("images_ds",
            ("dim_samples","dim_height_ds","dim_width_ds","dim_channels"),
            dtype=np.uint8, **var_params)
        var_img_ds.units = "Encoded R"
        var_time = create_var("time", ("dim_samples",), 
            chunksizes=(1,), dtype=np.float64, **var_params)
        var_time.units = "Seconds since 1970-01-01 00:00"

        var_img_ds[:] = images_ds
        var_img[:] = images_gen
        var_time[:] = time

    return (h, last_t)


def reconstruct_time_series_monthly(images_fn, weights_fn, out_dir,
    time_range, application="mchrzc", ds_factor=16, n_ensemble=4,
    relax_lam=0.0):

    (gen,_) = models.generator(num_timesteps=1)
    init_model = models.initial_state_model()
    (gen_init, noise_shapes) = models.generator_initialized(gen, init_model,
        num_timesteps=1)
    gen_init.load_weights(weights_fn)

    t0 = time_range[0]
    months = []
    while t0 < time_range[1]:
        (y,m) = (t0.year, t0.month)
        m += 1
        if m > 12:
            m = 1
            y += 1
        t1 = datetime(y,m,1)
        months.append((t0,t1))
        t0 = t1

    (h, last_t) = (None, None)
    for month in months:
        out_fn = out_dir + "/timeseries-{}-{}{:02d}.nc".format(
            application,month[0].year,month[0].month)
        (h, last_t) = reconstruct_time_series_partial(images_fn, gen,
            noise_shapes, init_model, out_fn, month, h=h, last_t=last_t,
            application=application, ds_factor=ds_factor, n_ensemble=n_ensemble,
            relax_lam=relax_lam
        )


def log_spectral_distance(img1, img2):
    def power_spectrum_dB(img):
        fx = np.fft.fft2(img)
        fx = fx[:img.shape[0]//2,:img.shape[1]//2]
        px = abs(fx)**2
        return 10 * np.log10(px)

    d = (power_spectrum_dB(img1)-power_spectrum_dB(img2))**2

    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))


def log_spectral_distance_batch(batch1, batch2):
    lsd_batch = []
    for i in range(batch1.shape[0]):
        for j in range(batch1.shape[1]):
            lsd = log_spectral_distance(
                batch1[i,j,:,:,0], batch2[i,j,:,:,0]
            )
            lsd_batch.append(lsd)
    return np.array(lsd_batch)


def image_quality(gen, batch_gen, noise_shapes, num_instances=1,
    N_batches=100):
    N = batch_gen.N
    #N_batches = N//batch_gen.batch_size
    img_shape = batch_gen.img_shape
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape),
        batch_size=batch_gen.batch_size, random_seed=1234)

    batch_gen.reset(random_seed=1234)
    rmse_all = []
    ssim_all = []
    lsd_all = []

    for k in range(N_batches):
        (img_real, img_ds) = next(batch_gen)
        for i in range(num_instances):
            n = noise_gen()
            img_gen = gen.predict([img_ds]+n)
            rmse = np.sqrt(((img_real-img_gen)**2).mean(axis=(2,3,4)))
            ssim = msssim.MultiScaleSSIM(img_real, img_gen, 1.0)
            lsd = log_spectral_distance_batch(img_real, img_gen)
            rmse_all.append(rmse.flatten())
            ssim_all.append(ssim.flatten())
            lsd_all.append(lsd.flatten())

    rmse_all = np.concatenate(rmse_all)
    ssim_all = np.concatenate(ssim_all)
    lsd_all = np.concatenate(lsd_all)

    return (rmse_all, ssim_all, lsd_all)


def quality_metrics_by_time(application, data_fn, out_fn,
    weights_dir, check_every=1):
    (wgan, batch_gen_train, batch_gen_valid, _,
        noise_shapes, steps_per_epoch) = train.setup_gan(data_fn,
            application=application, batch_size=32)
    gen = wgan.gen

    files = os.listdir(weights_dir)
    def get_app(fn):
        return fn.split("-")[1]
    files = sorted(fn for fn in files if get_app(fn)==application)

    def log_line(line):
        with open(out_fn, 'a') as f:
            print(line, file=f)
    log_line("N RMSE MSSSIM LSD")

    for fn in files[::check_every]:
        N_samples = int(fn.split("-")[-1].split(".")[0])
        print(N_samples)
        gen.load_weights(weights_dir+"/"+fn)

        (rmse, ssim, lsd) = image_quality(gen, batch_gen_valid, noise_shapes)
        log_line("{} {:.6f} {:.6f} {:.6f}".format(
            N_samples, rmse.mean(), ssim.mean(), np.nanmean(lsd)))


def quality_metrics_table(application, data_fn, weights_fn, method="gan"):
    if method == "gan":
        (wgan, batch_gen_train, batch_gen_valid, batch_gen_test,
            noise_shapes, steps_per_epoch) = train.setup_gan(data_fn,
                test_data_file=data_fn, application=application, batch_size=32)
        gen = wgan.gen
        gen.load_weights(weights_fn)
    elif method == "gen_det":
        (gen_det, batch_gen_train, batch_gen_valid, 
            batch_gen_test, steps_per_epoch) = train.setup_deterministic(data_fn,
            test_data_file=data_fn, sample_random=True, n_samples=1, batch_size=32,
            application=application, loss='mse')
        gen_det.load_weights(weights_fn)
        gen = GeneratorDeterministicPlaceholder(gen_det)
        noise_shapes = lambda s: []
    elif method == "lanczos":
        (gen_det, batch_gen_train, batch_gen_valid, 
            batch_gen_test, steps_per_epoch) = train.setup_deterministic(data_fn,
            test_data_file=data_fn, sample_random=True, n_samples=1, batch_size=32,
            application=application, loss='mse')
        gen = GeneratorLanczos((128,128))
        noise_shapes = lambda s: []
    elif method == "rainfarm":
        (gen_det, batch_gen_train, batch_gen_valid, 
            batch_gen_test, steps_per_epoch) = train.setup_deterministic(data_fn,
            test_data_file=data_fn, sample_random=True, n_samples=1, batch_size=32,
            application=application, loss='mse')
        gen = GeneratorRainFARM(16, batch_gen_test.decoder)
        noise_shapes = lambda s: []

    (rmse, ssim, lsd) = image_quality(gen, batch_gen_test, noise_shapes)

    print("RMSE: {:.3f}".format(rmse.mean()))
    print("MSSSIM: {:.3f}".format(ssim.mean()))
    print("LSD: {:.3f}".format(np.nanmean(lsd)))


class GeneratorLanczos:
    # class that can be used in place of a generator for evaluation purposes,
    # using Lanczos filtering
    def __init__(self, out_size):
        self.out_size = out_size

    def predict(self, *args):
        y = args[0][0]
        out_shape = y.shape[:2] + self.out_size + y.shape[4:]
        x = np.zeros(out_shape, dtype=y.dtype)
        for i in range(x.shape[0]):
            for k in range(x.shape[1]):
                x[i,k,:,:,0] = plots.resize_lanczos(y[i,k,:,:,0],
                    self.out_size)
        return x


class GeneratorDeterministicPlaceholder:
    def __init__(self, gen_det):
        self.gen_det = gen_det

    def predict(self, *args):
        y = args[0]
        return self.gen_det.predict(y)


class GeneratorRainFARM:
    def __init__(self, ds_factor, decoder):
        self.ds_factor = ds_factor
        self.decoder = decoder
        self.batches = 0

    def predict(self, *args):
        print(self.batches)
        self.batches += 1
        y = args[0][0]
        y = self.decoder.denormalize(y)
        P = 10**y
        P[~np.isfinite(P)] = 0

        out_size = (y.shape[2]*self.ds_factor, y.shape[3]*self.ds_factor)
        out_shape = y.shape[:2] + out_size + y.shape[4:]
        x = np.zeros(out_shape, dtype=y.dtype)

        for i in range(y.shape[0]):
            alpha = rainfarm.get_alpha_seq(P[i,...,0])
            r = [rainfarm.rainfarm_downscale(p, alpha=alpha, threshold=0.1, 
                ds_factor=self.ds_factor) for p in P[0,...,0]]
            log_r = np.log10(r)
            log_r[~np.isfinite(log_r)] = np.nan
            log_r = self.decoder.normalize(log_r)
            log_r[~np.isfinite(log_r)] = 0.0
            x[i,...,0] = log_r
            x = x.clip(0,1)

        return x
