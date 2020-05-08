import gc

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import generic_utils

from layers import GradientPenalty, RandomWeightedAverage
from meta import Nontrainable, input_shapes, ensure_list
from meta import save_opt_weights, load_opt_weights


class WGANGP(object):

    def __init__(self, gen, disc, num_channels=1, num_timesteps=8,
        gradient_penalty_weight=10, lr_disc=0.0001, lr_gen=0.0001,
        avg_seed=None):

        self.gen = gen
        self.disc = disc
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.build_wgan_gp()

    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5"
        }
        return fn

    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])
        
        with Nontrainable(self.disc):
            self.gen_trainer._make_train_function()
            load_opt_weights(self.gen_trainer,
                load_files["gen_opt_weights"])
        with Nontrainable(self.gen):
            self.disc_trainer._make_train_function()
            load_opt_weights(self.disc_trainer,
                load_files["disc_opt_weights"])


    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])


    def build_wgan_gp(self):

        # find shapes for inputs
        cond_shapes = input_shapes(self.gen, "cond")
        noise_shapes = input_shapes(self.gen, "noise")
        sample_shapes = input_shapes(self.disc, "sample")

        # Create generator training network
        with Nontrainable(self.disc):
            cond_in = [Input(shape=s) for s in cond_shapes]
            noise_in = [Input(shape=s) for s in noise_shapes]
            gen_in = cond_in+noise_in
            gen_out = self.gen(gen_in)
            gen_out = ensure_list(gen_out)
            disc_in_gen = cond_in+[gen_out]
            disc_out_gen = self.disc(disc_in_gen)
            self.gen_trainer = Model(inputs=gen_in, outputs=disc_out_gen)

        # Create discriminator training network
        with Nontrainable(self.gen):
            cond_in = [Input(shape=s) for s in cond_shapes]
            noise_in = [Input(shape=s) for s in noise_shapes]
            sample_in = [Input(shape=s) for s in sample_shapes]
            gen_in = cond_in+noise_in
            disc_in_real = sample_in[0]
            disc_in_fake = self.gen(gen_in) 
            disc_in_avg = RandomWeightedAverage()([disc_in_real,disc_in_fake])
            disc_out_real = self.disc(cond_in+[disc_in_real])
            disc_out_fake = self.disc(cond_in+[disc_in_fake])
            disc_out_avg = self.disc(cond_in+[disc_in_avg])
            disc_gp = GradientPenalty()([disc_out_avg, disc_in_avg])
            self.disc_trainer = Model(inputs=cond_in+sample_in+noise_in,
                outputs=[disc_out_real,disc_out_fake,disc_gp])

        self.compile()

    def compile(self, opt_disc=None, opt_gen=None):
        #create optimizers
        if opt_disc is None:
            opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_disc = opt_disc
        if opt_gen is None:
            opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)
        self.opt_gen = opt_gen

        with Nontrainable(self.disc):
            self.gen_trainer.compile(loss=wasserstein_loss,
                optimizer=self.opt_gen)
        with Nontrainable(self.gen):
            self.disc_trainer.compile(
                loss=[wasserstein_loss, wasserstein_loss, 'mse'], 
                loss_weights=[1.0, 1.0, self.gradient_penalty_weight],
                optimizer=self.opt_disc
            )

    def train(self, batch_gen, noise_gen, num_gen_batches=1, 
        training_ratio=1, show_progress=True):

        disc_target_real = None
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                num_gen_batches*batch_gen.batch_size)

        disc_target_real = np.ones(
            (batch_gen.batch_size, batch_gen.num_frames, 1), dtype=np.float32)
        disc_target_fake = -disc_target_real
        gen_target = disc_target_real
        target_gp = np.zeros((batch_gen.batch_size, 1), dtype=np.float32)
        disc_target = [disc_target_real, disc_target_fake, target_gp]

        loss_log = []

        for k in range(num_gen_batches):
        
            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            for rep in range(training_ratio):
                # generate some real samples
                (sample, cond) = next(batch_gen)
                noise = noise_gen()

                with Nontrainable(self.gen):   
                    dl = self.disc_trainer.train_on_batch(
                        [cond,sample]+noise, disc_target)

                if disc_loss is None:
                    disc_loss = np.array(dl)
                else:
                    disc_loss += np.array(dl)
                disc_loss_n += 1

                del sample, cond

            disc_loss /= disc_loss_n

            with Nontrainable(self.disc):
                (sample, cond) = next(batch_gen)
                gen_loss = self.gen_trainer.train_on_batch(
                    [cond]+noise_gen(), gen_target)
                del sample, cond

            if show_progress:
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                for (i,gl) in enumerate([gen_loss]):
                    losses.append(("G{}".format(i), gl))
                progbar.add(batch_gen.batch_size, 
                    values=losses)

            loss_log.append(np.hstack((disc_loss,gen_loss)))

            gc.collect()

        return np.array(loss_log)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)
