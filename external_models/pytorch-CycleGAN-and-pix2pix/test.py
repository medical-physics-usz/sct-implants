import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import pandas as pd
import numpy as np
import shutil
from glob import glob
import nibabel as nib

# --- PATH DEFINITION ---

# --- Constants ---

CT_THRESHOLDS = {
    "max": 4000,
    "min": -1024,
}

def create_directory(path):
    if os.path.exists(path):
        print("Such path {} exists. Removing it".format(path))
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
    os.makedirs(path)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # [thesis] input and output channel configuration added for thesis
    opt.input_modalities = [m.strip() for m in opt.input_modalities.split(',') if m.strip()]
    opt.output_modalities = [m.strip() for m in opt.output_modalities.split(',') if m.strip()]
    opt.input_nc = len(opt.input_modalities) + 2 * opt.pseudo3D_window * len(opt.input_modalities)
    opt.output_nc = len(opt.output_modalities)

    CT_THRESHOLDS["max"] = opt.max_ct_intensity

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of test images = %d' % len(dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers


    # Path for saving NIFTIs from inference
    path_results = opt.path_root_results
    create_directory(path_results)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


    # Running Model Inference

    # Model Setup and Initialization
    for i, data in enumerate(dataset):
        if i == 0:
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            if opt.eval:
                model.eval()

        # Image Processing
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results real_a, fake_B, real_B
        img_path = model.get_image_paths()  # get image paths


        # Transform back range [-1, 1] to HU range [CT-min, CT-max]
        ct_channel_index = opt.output_modalities.index('CT')
        fake_ct = ((visuals["fake_B"][:, ct_channel_index:ct_channel_index + 1, :, :] - (-1)) / 2) * (CT_THRESHOLDS["max"] - CT_THRESHOLDS["min"]) + CT_THRESHOLDS["min"]
        fake_ct_numpy = fake_ct[0].clamp(CT_THRESHOLDS["min"], CT_THRESHOLDS["max"]).cpu().float().numpy().astype(np.int16).squeeze()

        #  Extracting slice number and patient number (PatXXX)
        file_name = str(img_path[0][0].split('/')[-1])
        patient_nr, slice_nr = file_name.split('-')
        slice_nr = slice_nr.split('.')[0]

        # Save model inferred sCT images to NIFTI files
        path_fake_nifti = os.path.join(path_results, "fake_nifti", patient_nr)
        os.makedirs(path_fake_nifti, exist_ok=True)
        path_slice_fake = os.path.join(path_fake_nifti, patient_nr + "_" + slice_nr + '.nii')
        fake_ct_nif = nib.Nifti1Image(fake_ct_numpy, np.eye(4))
        nib.save(fake_ct_nif, path_slice_fake)














