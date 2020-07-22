import argparse
import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD

import plots
import train


path = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="train or plot")
    parser.add_argument('--data_file', type=str, 
        help="Training data file")
    parser.add_argument('--test_data_file', type=str, default="",
        help="Test data file")
    parser.add_argument('--application', type=str, default="mchrzc",
        help="Network weights file root to load")
    parser.add_argument('--load_weights_root', type=str, default="",
        help="Network weights file root to load")
    parser.add_argument('--save_weights_root', type=str, default="",
        help="Network weights file root to save")
    parser.add_argument('--log_path', type=str, default="",
        help="Log files path")
    parser.add_argument('--steps_per_epoch', type=int, default=200,
        help="Batches per epoch")
    parser.add_argument('--batch_size', type=int, default=16,
        help="Batch size")
    parser.add_argument('--num_samples', type=int, default=400000,
        help="Training samples")
    parser.add_argument('--opt_switch_point', type=int, default=350000,
        help="The num. of samples at which the optimizer is switched to SGD")
    parser.add_argument('--mchrzc_data_file', type=str, default="",
        help="MCH-RZC data file")
    parser.add_argument('--goescod_data_file', type=str, default="",
        help="GOES-COT data file")
        
    args = parser.parse_args()
    mode = args.mode

    if mode=="train":

        data_fn = args.data_file
        application = args.application
        load_weights_root = args.load_weights_root
        save_weights_root = args.save_weights_root
        log_path = args.log_path
        steps_per_epoch = args.steps_per_epoch
        batch_size = args.batch_size
        num_samples = args.num_samples
        opt_switch_point = args.opt_switch_point

        if not save_weights_root:
            save_weights_root = path + "../models/downscaleseqgan"

        # initialize GAN
        (wgan, batch_gen_train, batch_gen_valid, _, noise_shapes, _) = \
            train.setup_gan(data_fn, 
                batch_size=batch_size, application=application)

        if load_weights_root: # load weights and run status
            wgan.load(wgan.filenames_from_root(load_weights_root))
            with open(load_weights_root+"-run_status.json", 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path, application)
                log = pd.read_csv(log_file)

        else: # initialize run status
            chars = string.ascii_lowercase + string.digits
            training_samples = 0

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path, application)
                log = pd.DataFrame(columns=["training_samples", 
                    "disc_loss", "disc_loss_real", "disc_loss_fake",
                    "disc_loss_gp", "gen_loss"])

        plot_fn = "{}/progress-{}.pdf".format(log_path,application) if log_path \
            else path+"/../figures/progress.pdf"
        switched_opt = (training_samples >= opt_switch_point)

        while (training_samples < num_samples): # main training loop

            # check if we should switch optimizers
            if (training_samples >= opt_switch_point) and not switched_opt:
                opt_disc = SGD(1e-5)
                opt_gen = SGD(1e-5)
                wgan.compile(opt_disc=opt_disc, opt_gen=opt_gen)
                switched_opt = True

            # train for some number of batches
            loss_log = train.train_gan(wgan, batch_gen_train,
                batch_gen_valid, noise_shapes,
                steps_per_epoch, 1, plot_fn=plot_fn)
            loss_log = np.mean(loss_log, axis=0)
            training_samples += steps_per_epoch * batch_gen_train.batch_size

            # save results
            wgan.save(save_weights_root)
            run_status = {
                "application": application,
                "training_samples": training_samples,
            }
            with open(save_weights_root+"-run_status.json", 'w') as f:
                json.dump(run_status, f)

            if log_path: # log losses and generator weights for evaluation
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "disc_loss": [loss_log[0]],
                    "disc_loss_real": [loss_log[1]],
                    "disc_loss_fake": [loss_log[2]],
                    "disc_loss_gp": [loss_log[3]],
                    "gen_loss": [loss_log[4]]
                }))
                log.to_csv(log_file, index=False, float_format="%.6f")
                        
                gen_weights_file = "{}/gen_weights-{}-{:07d}.h5".format(
                    log_path, application, training_samples)
                wgan.gen.save_weights(gen_weights_file)


    elif mode == "plot":
        mchrzc_data_fn = args.mchrzc_data_file
        goescod_data_fn = args.goescod_data_file

        plots.plot_all(mchrzc_data_fn, goescod_data_fn)
