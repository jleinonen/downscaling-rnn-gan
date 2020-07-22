from bisect import bisect_left
from datetime import datetime, timedelta
import gc
import os
from string import ascii_lowercase

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
import netCDF4
import numpy as np
import pandas as pd
try:
    from PIL import Image
except ImportError:
    pass # to allow loading on setups witout PIL

import data
import models
import noise
import train


path = os.path.dirname(os.path.abspath(__file__))


def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.Normalize(*value_range), extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_sequences(gen, batch_gen, noise_gen, 
    num_samples=8, num_instances=4, out_fn=None,
    plot_stride=1):

    old_batch_size = batch_gen.batch_size
    try:
        batch_gen.batch_size = num_samples
        noise_gen.batch_size = num_samples
        (seq_real, cond) = next(batch_gen)
        seq_gen = []
        for i in range(num_instances):
            seq_gen.append(gen.predict([cond]+noise_gen()))
    finally:
        batch_gen.batch_size = old_batch_size
        noise_gen.batch_size = old_batch_size

    seq_real = batch_gen.decoder.denormalize(seq_real)
    cond = batch_gen.decoder.denormalize(cond)
    seq_gen = [batch_gen.decoder.denormalize(seq) for seq in seq_gen]

    num_frames = batch_gen.num_frames
    if plot_stride > 1:
        seq_real = seq_real[:,::plot_stride,...]
        cond = cond[:,::plot_stride,...]
        for i in range(len(seq_gen)):
            seq_gen[i] = seq_gen[i][:,::plot_stride,...]
        num_frames = seq_real.shape[1]

    num_rows = num_samples*num_frames
    num_cols = 2+num_instances

    figsize = (num_cols*1.5, num_rows*1.5)
    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows, num_cols, 
        wspace=0.05, hspace=0.05)

    value_range = batch_gen.decoder.value_range

    for s in range(num_samples):
        for t in range(num_frames):
            i = s*num_frames+t
            plt.subplot(gs[i,0])
            plot_img(seq_real[s,t,:,:,0], value_range=value_range)
            plt.subplot(gs[i,1])
            plot_img(cond[s,t,:,:,0], value_range=value_range)
            for k in range(num_instances):
                j = 2+k
                plt.subplot(gs[i,j])
                plot_img(seq_gen[k][s,t,:,:,0], value_range=value_range) 
            
    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def plot_rank_metrics_by_samples(metrics_fn,ax=None,
    plot_metrics=["KS", "DKL", "OP", "mean"], value_range=(-0.1,0.2),
    linestyles=['solid', 'dashed', 'dashdot', ':',],
    opt_switch_point=350000, plot_switch_text=True):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(metrics_fn, delimiter=" ")

    x = df["N"]
    for (metric,linestyle) in zip(plot_metrics,linestyles):
        y = df[metric]
        label = metric
        if metric=="DKL":
            label = "$D_\\mathrm{KL}$"
        if metric=="OP":
            label = "OF"
        if metric=="mean":
            y = y-0.5
            label = "mean - $\\frac{1}{2}$"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0,x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    ax.axvline(opt_switch_point, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    if plot_switch_text:
        text_x = opt_switch_point*0.98
        text_y = value_range[1]-(value_range[1]-value_range[0])*0.02
        ax.text(text_x, text_y, "Adam\u2192SGD", horizontalalignment='right',
            verticalalignment='top', color=(0.5,0.5,0.5))
    plt.grid(axis='y')


def plot_rank_metrics_by_samples_multiple(metrics_files,
    value_ranges=[(-0.025,0.075),(-0.1,0.2)]):
    (fig,axes) = plt.subplots(len(metrics_files),1, sharex=True,
        squeeze=True)
    plt.subplots_adjust(hspace=0.1)

    for (i,(ax,fn,vr)) in enumerate(zip(axes,metrics_files,value_ranges)):
        plot_rank_metrics_by_samples(fn,ax,plot_switch_text=(i==0),value_range=vr)
        if i==len(metrics_files)-1:
            ax.legend(ncol=5)
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
        ax.set_ylabel("Rank metric")
        ax.grid(axis='y')


def plot_quality_metrics_by_samples(quality_metrics_fn,
    rank_metrics_fn, ax=None,
    plot_metrics=["RMSE", "MSSSIM", "LSD", "CRPS"], value_range=(0,0.7),
    linestyles=['-', '--', ':', '-.'], opt_switch_point=350000,
    plot_switch_text=True):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(quality_metrics_fn, delimiter=" ")
    df_r = pd.read_csv(rank_metrics_fn, delimiter=" ")
    df["CRPS"] = df_r["CRPS"]

    x = df["N"]
    for (metric,linestyle) in zip(plot_metrics,linestyles):
        y = df[metric]
        label = metric
        if metric=="MSSSIM":
            y = 1-y
            label = "$1 - $MS-SSIM"
        if metric=="LSD":
            label = "LSD [dB] / 50"
            y = y/50
        if metric=="CRPS":
            y = y*10
            label = "CRPS $\\times$ 10"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0,x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    ax.axvline(opt_switch_point, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    if plot_switch_text:
        text_x = opt_switch_point*0.98
        text_y = value_range[1]-(value_range[1]-value_range[0])*0.02
        ax.text(text_x, text_y, "Adam\u2192SGD", horizontalalignment='right',
            verticalalignment='top', color=(0.5,0.5,0.5))


def plot_quality_metrics_by_samples_multiple(
    quality_metrics_files, rank_metrics_files):

    (fig,axes) = plt.subplots(len(quality_metrics_files),1, sharex=True,
        squeeze=True)
    plt.subplots_adjust(hspace=0.1)
    value_ranges = [(0,0.4),(0,0.8)]

    for (i,(ax,fn_q,fn_r,vr)) in enumerate(zip(
        axes,quality_metrics_files,rank_metrics_files,value_ranges)):
        plot_quality_metrics_by_samples(fn_q,fn_r,ax,
            plot_switch_text=(i==0), value_range=vr)
        if i==0:
            ax.legend(mode='expand', ncol=4, loc='lower left')
        if i==1:
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
        ax.set_ylabel("Quality metric")
        ax.grid(axis='y')


def plot_sequences_horiz(gen, noise_shapes, batch_gen,
    samples=[0,1,2], num_instances=3, out_fn=None,
    plot_stride=2, random_seed=1234, application="mchrzc"):

    num_samples = len(samples)
    old_batch_size = batch_gen.batch_size
    old_augment = batch_gen.augment
    old_zeros_frac = batch_gen.zeros_frac
    img_shape = batch_gen.sequences.shape[2:4]
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape),
        batch_size=num_samples, random_seed=random_seed)
    # force the batch generator to return the selected samples
    batch_gen.next_ind = np.array(samples)
    try:
        batch_gen.batch_size = num_samples
        batch_gen.augment = False
        batch_gen.zeros_frac = 0.0
        (seq_real, cond) = next(batch_gen)
        seq_gen = []
        for i in range(num_instances):
            seq_gen.append(gen.predict([cond]+noise_gen()))
    finally:
        batch_gen.batch_size = old_batch_size
        batch_gen.augment = old_augment
        batch_gen.zeros_frac = old_zeros_frac

    seq_real = batch_gen.decoder.denormalize(seq_real)
    cond = batch_gen.decoder.denormalize(cond)
    seq_gen = [batch_gen.decoder.denormalize(seq) for seq in seq_gen]

    num_frames = batch_gen.num_frames
    if plot_stride > 1:
        seq_real = seq_real[:,::plot_stride,...]
        cond = cond[:,::plot_stride,...]
        for i in range(len(seq_gen)):
            seq_gen[i] = seq_gen[i][:,::plot_stride,...]
        num_frames = seq_real.shape[1]

    num_rows = num_samples
    num_cols = num_frames
    num_rows_s = 2+num_instances

    figsize = (num_cols*1.5, num_rows*num_rows_s*1.60)
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows+1, 1, hspace=0.05,
        height_ratios=[1]*num_rows+[0.035])

    value_range = batch_gen.decoder.value_range

    for s in range(num_samples):
        gs_s = gridspec.GridSpecFromSubplotSpec(num_rows_s, num_cols,
            subplot_spec=gs[s,0], wspace=0.05, hspace=0.05)
        for t in range(num_frames):
            plt.subplot(gs_s[0,t])
            plot_img(seq_real[s,t,:,:,0], value_range=value_range)
            if t==0:
                plt.ylabel("Real", fontsize=16)
                plt.text(0.01, 0.97, "({})".format(ascii_lowercase[s]),
                    horizontalalignment='left', verticalalignment='top',
                    transform=plt.gca().transAxes, fontsize=16)
                if s==0:
                    plt.title("Time \u2192", fontsize=16)
            plt.subplot(gs_s[1,t])
            plot_img(cond[s,t,:,:,0], value_range=value_range)
            if t==0:
                plt.ylabel("Downs.", fontsize=16)
            for k in range(num_instances):
                j = 2+k
                plt.subplot(gs_s[j,t])
                plot_img(seq_gen[k][s,t,:,:,0], value_range=value_range) 
                if t==0:
                    plt.ylabel("Gen. #{}".format(k+1), fontsize=16)

    if application == 'mchrzc':
        units = "Rain rate [mm h$^{-1}$]"
        cb_tick_loc = np.array([-1, 0, 1, 2])
        cb_tick_labels = [0.1, 1, 10, 100]
    elif application == 'goescod':
        units = "Cloud optical thickness"
        cb_tick_loc = np.log([2, 10, 50, 150])
        cb_tick_labels = np.exp(cb_tick_loc).round().astype(int)
            
    cax = plt.subplot(gs[-1,0]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.Normalize(*value_range),
        orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=16)
    cb.set_label(units, size=16)

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def plot_examples_mchrzc(data_fn, weights_fn, plot_fn):
    (wgan, batch_gen_train, batch_gen_valid, batch_gen_test, noise_shapes,
        steps_per_epoch) = train.setup_gan(data_fn, test_data_file=data_fn,
        sample_random=True, n_samples=1, application='mchrzc',
        random_seed=1234)
    gen = wgan.gen
    gen.load_weights(weights_fn)
    plot_sequences_horiz(gen, noise_shapes, batch_gen_test, samples=[0,21,15],
        application='mchrzc', plot_stride=1)
    plt.savefig(plot_fn, bbox_inches='tight')
    plt.close()


def plot_examples_goescod(data_fn, weights_fn, plot_fn):
    (wgan, batch_gen_train, batch_gen_valid, batch_gen_test, noise_shapes,
        steps_per_epoch) = train.setup_gan(data_fn, test_data_file=data_fn,
        sample_random=True, n_samples=1, application='goescod',
        random_seed=1234)
    gen = wgan.gen
    gen.load_weights(weights_fn)
    plot_sequences_horiz(gen, noise_shapes, batch_gen_test, samples=[0,1,2],
        application='goescod', plot_stride=1)
    plt.savefig(plot_fn, bbox_inches='tight')
    plt.close()


def plot_examples_mchrzc_random(data_fn, weights_fn, plot_dir, num_examples=16):
    (wgan, batch_gen_train, batch_gen_valid, batch_gen_test, noise_shapes,
        steps_per_epoch) = train.setup_gan(data_fn, test_data_file=data_fn,
        sample_random=True, n_samples=1, application='mchrzc',
        random_seed=2345)
    gen = wgan.gen
    gen.load_weights(weights_fn)
    for k in range(num_examples):
        plot_fn = plot_dir + "/examples-mchrzc-random-{:02d}.pdf".format(k)
        plot_sequences_horiz(gen, noise_shapes, batch_gen_test, samples=[k],
            application='mchrzc', plot_stride=1, num_instances=12)
        plt.savefig(plot_fn, bbox_inches='tight')
        plt.close()


def plot_examples_goescod_random(data_fn, weights_fn, plot_dir, num_examples=16):
    (wgan, batch_gen_train, batch_gen_valid, batch_gen_test, noise_shapes,
        steps_per_epoch) = train.setup_gan(data_fn, test_data_file=data_fn,
        sample_random=True, n_samples=1, application='goescod',
        random_seed=2345)
    gen = wgan.gen
    gen.load_weights(weights_fn)
    for k in range(num_examples):
        plot_fn = plot_dir + "/examples-goescod-random-{:02d}.pdf".format(k)
        plot_sequences_horiz(gen, noise_shapes, batch_gen_test, samples=[k],
            application='goescod', plot_stride=1, num_instances=12)
        plt.savefig(plot_fn, bbox_inches='tight')
        plt.close()


def plot_video_frame(img_real, img_ds, img_gen, oob_mask, time, num_ensemble=4):
    assert(num_ensemble in {1,4})

    img_shape = img_real.shape

    if num_ensemble == 1:
        figsize = (img_shape[1]/img_shape[0]*3*4, 4)
        gs = gridspec.GridSpec(1,3,hspace=0.05, wspace=0.05)
    elif num_ensemble == 4:
        figsize = (img_shape[1]/img_shape[0]*3*4, 2*4)
        gs = gridspec.GridSpec(2,3,hspace=0.05, wspace=0.05)

    fig = plt.figure(figsize=figsize, dpi=210)
    ds_factor = int(round((img_gen.shape[0]/img_ds.shape[0])))

    oob_mask_ds = np.zeros(img_ds.shape, dtype=bool)
    oob_mask_gen = np.zeros(img_gen[:,:,0].shape, dtype=bool)
    for i_ds in range(oob_mask_ds.shape[0]):
        for j_ds in range(oob_mask_ds.shape[1]):
            i0 = i_ds*ds_factor
            j0 = j_ds*ds_factor
            i1 = i0+ds_factor
            j1 = j0+ds_factor
            oob_mask_ds[i_ds,j_ds] = oob_mask[i0:i1,j0:j1].any()
            oob_mask_gen[i0:i1,j0:j1] = oob_mask_ds[i_ds,j_ds]
    cmap_mask = colors.ListedColormap([
        [0.0,0.0,0.0,0.0],
        [0.75,0.75,0.75,1.0]
    ])

    import shapefile
    border = shapefile.Reader("../data/Border_CH.shp")
    shapes = list(border.shapeRecords())
    def draw_border():
        for shape in shapes:
            x = [i[0]/1000. for i in shape.shape.points[:]]
            y = [i[1]/1000. for i in shape.shape.points[:]]
            plt.plot(x,y,'k',linewidth=1.0)
    extent_real = [254.5,965.5,-159.5,480.5]
    extent_gen = [254.5,959.5,-159.5,480.5]

    plt.subplot(gs[0,0])
    plot_img(img_real, extent=extent_real)
    plt.imshow(oob_mask.astype(int), cmap=cmap_mask, extent=extent_real)
    draw_border()
    plt.gca().set_xlim((extent_real[0],extent_real[1]))
    plt.gca().set_ylim((extent_real[2],extent_real[3]))
    plt.title("Real", fontsize=14)

    if num_ensemble == 1:
        gs_ds = gs[0,1]
    elif num_ensemble == 4:
        gs_ds = gs[1,0]
    plt.subplot(gs_ds)
    plot_img(img_ds, extent=extent_gen)
    plt.imshow(oob_mask_ds.astype(int), cmap=cmap_mask, extent=extent_gen)
    draw_border()
    plt.gca().set_xlim((extent_real[0],extent_real[1]))
    plt.gca().set_ylim((extent_real[2],extent_real[3]))
    
    if num_ensemble == 1:
        plt.title(time.strftime("%Y-%m-%d %H:%M UTC")+"\n\nDownsampled", fontsize=14)
    elif num_ensemble == 4:
        plt.xlabel("Downsampled", fontsize=14)

    if num_ensemble == 1:
        gs_list = [gs[0,2]]
    elif num_ensemble == 4:
        gs_list = [gs[0,1], gs[0,2], gs[1,1], gs[1,2]]
    for (k,g) in enumerate(gs_list):
        plt.subplot(g)
        plot_img(img_gen[:,:,k], extent=extent_gen)
        plt.imshow(oob_mask_gen.astype(int), cmap=cmap_mask, extent=extent_gen)
        draw_border()
        plt.gca().set_xlim((extent_real[0],extent_real[1]))
        plt.gca().set_ylim((extent_real[2],extent_real[3]))

        if num_ensemble == 1:
            plt.title("Reconstructed", fontsize=14)
        elif num_ensemble == 4:
            if k == 1:
                plt.title("Generated #{}".format(k+1), fontsize=14)
            elif k == 0:
                plt.title(time.strftime("%Y-%m-%d %H:%M UTC") +
                    "\n\nGenerated#{}".format(k+1), fontsize=14)
            else:
                plt.xlabel("Generated #{}".format(k+1), fontsize=14)



def plot_video_frames_all(images_fn, gen_fn, out_dir,
    format="png", application="mchrzc", time_range=None,
    scaling_fn=path+"/../data/scale_rzc.npy", num_ensemble=4):

    if application == "mchrzc":
        dec = data.RainRateDecoder(scaling_fn, below_val=np.log10(0.025))
    else:
        raise ValueError("Unknown application.")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    smoothener = data.Smoothener()

    # To get a proper comparison of real and generated fields we must
    # apply the same kind of preprocessing as we do when training the GAN
    def decode_real(x): 
        oob_mask = (x==0)
        nan_mask = (x==1)
        x = dec(x)
        x[nan_mask | oob_mask] = dec.below_val
        x = x.reshape((1,1)+img_real.shape+(1,))
        x = smoothener.smoothen(x)
        x = x[0,0,:,:,0]
        x[x < dec.value_range[0]] = np.nan
        return (x, oob_mask)

    def decode(x):
        oob_mask = (x==0)
        nan_mask = (x==1)
        x = dec(x)
        x[nan_mask | oob_mask] = np.nan
        return x

    with netCDF4.Dataset(images_fn, 'r') as ds_images:
        time_real = np.array(ds_images["time"][:], copy=False)
        with netCDF4.Dataset(gen_fn, 'r') as ds_gen:
            t0 = ds_gen["time"][0]
            k0 = bisect_left(time_real,t0)

            N = ds_gen["images"].shape[0]
            for k in range(k0,k0+ds_gen["images"].shape[0]):
                time = float(ds_images["time"][k])
                time = datetime(1970,1,1)+timedelta(seconds=time)
                if time_range is not None:
                    if not (time_range[0]<=time<time_range[1]):
                        continue
                print(k)
                img_real = np.array(ds_images["images"][k,:,:,0], copy=False)
                (img_real,oob_mask) = decode_real(img_real)
                img_ds = decode(np.array(ds_gen["images_ds"][k-k0,:,:,0], copy=False))
                img_gen = decode(np.array(ds_gen["images"][k-k0,:,:,0,:], copy=False))

                plot_video_frame(img_real, img_ds, img_gen,
                    oob_mask, time, num_ensemble=num_ensemble)
                out_fn = "{}/frame-{:05d}.{}".format(out_dir,k,format)
                plt.savefig(out_fn, bbox_inches='tight')
                plt.close()


def plot_rank_histogram(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0,1,N_ranks)
    db = (bc[1]-bc[0])
    bins = bc-db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h,_) = np.histogram(ranks,bins=bins)
    h = h / h.sum()

    ax.plot(bc,h,**plot_params)


def plot_rank_cdf(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0,1,N_ranks)
    db = (bc[1]-bc[0])
    bins = bc-db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h,_) = np.histogram(ranks,bins=bins)
    h = h.cumsum()
    h = h / h[-1]

    ax.plot(bc,h,**plot_params)


def plot_rank_histogram_all(rank_files, labels, N_ranks=101):
    (fig,axes) = plt.subplots(2,1,sharex=True,figsize=(6,3))
    plt.subplots_adjust(hspace=0.15)

    linestyles = ["-","--"]
    colors = ["C0", "C1"]

    for ((fn_valid,fn_test),label,ls,c) in zip(rank_files,labels,linestyles,colors):
        with np.load(fn_test, allow_pickle=True) as f:
            ranks = f['arr_0'].item()['ranks']
        plot_rank_histogram(axes[0], ranks, N_ranks=N_ranks,
            label=label, linestyle=ls, linewidth=2, c=c, alpha=0.7, zorder=1)
        with np.load(fn_valid) as f:
            ranks = f['arr_0']
        plot_rank_histogram(axes[0], ranks, N_ranks=N_ranks,
            label=None, linestyle=ls, linewidth=0.75, c=c, zorder=2)
    bc = np.linspace(0,1,N_ranks)
    axes[0].plot(bc, [1./N_ranks]*len(bc), linestyle=':', label="Uniform", c='C2', zorder=0)
    axes[0].set_ylabel("Norm. occurrence")
    ylim = axes[0].get_ylim()
    axes[0].set_ylim((0,ylim[1]))
    axes[0].set_xlim((0,1))
    axes[0].text(0.01, 0.97, "(a)",
        horizontalalignment='left', verticalalignment='top',
        transform=axes[0].transAxes)

    for ((fn_valid,fn_test),label,ls,c) in zip(rank_files,labels,linestyles,colors):
        with np.load(fn_test, allow_pickle=True) as f:
            ranks = f['arr_0'].item()['ranks']
        plot_rank_cdf(axes[1], ranks, N_ranks=N_ranks,
            label=label, linestyle=ls, linewidth=2, c=c, alpha=0.7, zorder=1)
        with np.load(fn_valid) as f:
            ranks = f['arr_0']
        plot_rank_cdf(axes[1], ranks, N_ranks=N_ranks,
            label=None, linestyle=ls, linewidth=0.75, c=c, zorder=2)
    axes[1].plot(bc,bc,linestyle=':', label="Uniform", c='C2', zorder=0)
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("Normalized rank")
    axes[1].set_ylim(0,1)
    axes[1].set_xlim((0,1))
    axes[1].text(0.01, 0.97, "(b)",
        horizontalalignment='left', verticalalignment='top',
        transform=axes[1].transAxes)
    axes[1].legend(loc='lower right')


def plot_all(
        mchrzc_data_fn,
        goescod_data_fn,
        figs_dir="../figures/",
        mchrzc_gen_weights_fn="../models/gen_weights-mchrzc-0361600.h5",
        goescod_gen_weights_fn="../models/gen_weights-goescod-0371200.h5",
        mchrzc_quality_metrics_fn="../data/quality_metrics_by_time-mchrzc.txt",
        goescod_quality_metrics_fn="../data/quality_metrics_by_time-goescod.txt",
        mchrzc_rank_metrics_fn="../data/rank_metrics_by_time-mchrzc.txt",
        goescod_rank_metrics_fn="../data/rank_metrics_by_time-goescod.txt",
        mchrzc_rank_samples_valid_fn="../data/ranks-mchrzc-361600-valid.npz",
        mchrzc_rank_samples_test_fn="../data/ranks-mchrzc-361600-test.npz",
        goescod_rank_samples_valid_fn="../data/ranks-goescod-371200-valid.npz",
        goescod_rank_samples_test_fn="../data/ranks-goescod-371200-test.npz"
    ):

    plot_examples_mchrzc(
        mchrzc_data_fn,
        mchrzc_gen_weights_fn,
        "{}/examples-mchrzc.pdf".format(figs_dir)
    )
    gc.collect()

    plot_examples_mchrzc_random(
        mchrzc_data_fn, mchrzc_gen_weights_fn, figs_dir
    )
    gc.collect()

    plot_examples_goescod(
        goescod_data_fn,
        goescod_gen_weights_fn,
        "{}/examples-goescod.pdf".format(figs_dir)
    )
    gc.collect()

    plot_examples_goescod_random(
        goescod_data_fn, goescod_gen_weights_fn, figs_dir
    )
    gc.collect()

    plot_quality_metrics_by_samples_multiple(
        [mchrzc_quality_metrics_fn, goescod_quality_metrics_fn], 
        [mchrzc_rank_metrics_fn, goescod_rank_metrics_fn]
    )
    plt.savefig("{}/quality-metrics-time.pdf".format(figs_dir),
        bbox_inches='tight')
    plt.close()

    plot_rank_metrics_by_samples_multiple(
        [mchrzc_rank_metrics_fn, goescod_rank_metrics_fn]
    )
    plt.savefig("{}/rank-metrics-time.pdf".format(figs_dir),
        bbox_inches='tight')
    plt.close()

    plot_rank_histogram_all(
        [
            (mchrzc_rank_samples_valid_fn,mchrzc_rank_samples_test_fn), 
            (goescod_rank_samples_valid_fn,goescod_rank_samples_test_fn), 
        ],
        ["MCH-RZC", "GOES-COT"]
    )
    plt.savefig("{}/rank-distribution.pdf".format(figs_dir),
        bbox_inches='tight')
    plt.close()

    plots.plot_comparison("/data/nowgan/test-samples-2017-128x128.nc", 
        "../models/gen_weights-mchrzc-0361600.h5", 
        "../models/gen_det_weights-mse.h5", random_seed=16)
    plt.savefig("../figures/comparison.pdf", bbox_inches='tight')
    plt.close()


def resize_lanczos(img, size):
    return np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))


def plot_comparison(test_data_file, gen_gan_weights, gen_det_mse_weights,
    application="mchrzc", random_seed=None):

    (_, _, batch_gen) = train.setup_batch_gen(
        test_data_file, test_data_file=test_data_file,
        application=application, random_seed=random_seed,
        batch_size=1
    )

    old_batch_size = batch_gen.batch_size
    try:
        batch_gen.batch_size = 1
        (seq_real, cond) = next(batch_gen)
    finally:
        batch_gen.batch_size = old_batch_size

    size = tuple(seq_real.shape[2:4])
    seq_lanczos = np.array([resize_lanczos(x, size) for x in cond[0,...,0]])

    (gen, _) = models.generator()
    init_model = models.initial_state_model()
    (gen_gan, noise_shapes) = models.generator_initialized(
        gen, init_model)
    gen_det = models.generator_deterministic(gen_gan)

    noise = [np.random.randn(*((1,)+s)) for s in noise_shapes(size)]
    gen_gan.load_weights(gen_gan_weights)
    seq_gan = gen_gan.predict([cond]+noise)
    gen_det.load_weights(gen_det_mse_weights)
    seq_mse = gen_det.predict(cond)

    seq_real = batch_gen.decoder.denormalize(seq_real)
    cond = batch_gen.decoder.denormalize(cond)
    seq_lanczos = batch_gen.decoder.denormalize(seq_lanczos)
    seq_mse = batch_gen.decoder.denormalize(seq_mse)
    seq_gan = batch_gen.decoder.denormalize(seq_gan)

    import rainfarm
    P = 10**cond
    P[~np.isfinite(P)] = 0
    alpha = rainfarm.get_alpha_seq(P[0,...,0])
    print(alpha)
    r = [rainfarm.rainfarm_downscale(p, alpha=alpha, threshold=0.1)
        for p in P[0,...,0]]
    log_r = np.log10(r)
    log_r[~np.isfinite(log_r)] = np.nan

    sequences = [
        seq_real[0,...,0],
        cond[0,...,0],
        seq_lanczos,
        seq_mse[0,...,0],
        log_r,
        seq_gan[0,...,0]
    ]
    labels = [
        "Real", "Downsampled", "Lanczos", "Det. RCNN", "RainFARM", "GAN"
    ]

    num_cols = seq_real.shape[1]
    num_rows = len(sequences)
    plt.figure(figsize=(1.5*num_cols,1.5*num_rows))

    gs = gridspec.GridSpec(num_rows,num_cols,wspace=0.05,hspace=0.05)
    
    for k in range(seq_real.shape[1]):
        for i in range(num_rows):
            plt.subplot(gs[i,k])
            plot_img(sequences[i][k,:,:])
            if k==0:
                plt.ylabel(labels[i])

    gc.collect()
