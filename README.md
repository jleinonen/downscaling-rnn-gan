# Stochastic, Recurrent Super-Resolution GAN for Downscaling Time-Evolving Atmospheric Fields

This is a reference implementation of a stochastic, recurrent super-resolution GAN for downscaling time-evolving fields, intended for use in the weather and climate sciences domain. This code supports a paper to be submitted to IEEE Transactions in Geoscience and Remote Sensing.

## Obtaining the data

The radar precipitation dataset (MCH-RZC in the paper) can be downloaded at https://doi.org/10.7910/DVN/ZDWWMG by following the instructions there. The GOES cloud optical thickness dataset (GOES-COT) can be found [in this data repository](https://doi.org/10.5281/zenodo.3835849) as "goes-samples-2019-128x128.nc".

## Obtaining the trained network

The trained generator weights selected for use in the paper are included in the `models` directory. The weights for the other time steps can be found [here](https://doi.org/10.5281/zenodo.3835849).

## Running the code

For training, you'll want a machine with a GPU and around 32 GB of memory (the training procedure for the radar dataset loads the entire dataset into memory). Running the pre-trained model should work just fine on a CPU.

You may want to work with the code interactively; in this case, just start a Python shell in the `dsrnngan` directory.

If you want the simplest way to run the code, the following two options are available. You may also want to look at what `main.py` does in order to get an idea of how the training/plotting flow works.

### Producing plots

You can replicate the plots in the paper (except for Fig. 7 for which we unfortunately cannot release the source data) by going to the `dsrnngan` directory and using
```
python main.py plot --mchrzc_data_file=<mchrzc_data_file> --goescod_data_file=<<mchrzc_data_file>>
```
where `<mchrzc_data_file>` is the path to the radar precipitation dataset and `<<mchrzc_data_file>` is the path to the GOES cloud optical thickness dataset. For more control over the plotting process, see the function `plot_all` in `plots.py`.

### Training the model

Run the following to start the training:
```
python main.py train --application=<application> --data_file=<data_file> --save_weights_root=<save_weights_root> --log_path=<log_path>
```
where `<application>` is either `mchrzc` (for the MCH-RZC dataset) or `goescod` (for the GOES-COT dataset), `<data_file>` is the training data file appropriate for the application, `<save_weights_root>` is the directory and file name root to which you want to save all model weights, and `<log_path>` is a path to a log directory where the logs and generator weights over time will be saved. 

The above command will run the training loop for 400000 generator training sequences and save the weights after each 3200 sequences.
