"""Running the motion simulation

Run this script with:

    python -u Simulate_Motion_Whole_Dataset.py --config_path path_to_config_file.yaml

Note: A config file needs to be generated first. This can be done manually or using the script 'Prepare_config.py'.
Examples of a config file for simulating train and validation data can be found under config/.
"""

import glob
import os.path
import numpy as np
from Utils import load_all_echoes, SimulateMotionForEachReadout, ImportMotionDataPCAfMRI, transform_sphere, ImportMotionDataNpy
import nibabel as nib
import time
import h5py
import argparse
import yaml
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ff506e', '#69005f', 'tab:gray', 'tab:olive', 'tab:blue', 'peru'])


parser = argparse.ArgumentParser(description='Motion simulation')
parser.add_argument('--config_path',
                    type=str,
                    default='config/example_config_train.yaml',
                    metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config_file = yaml.load(stream_file, Loader=yaml.FullLoader)

total = len(config_file['subject'])
if isinstance(config_file['simulation_nr'], list):
    total = len(config_file['subject']) * len(config_file['simulation_nr'])

current = 1
for sub, task, dataset in zip(config_file['subject'], config_file['task'], config_file['dataset']):

    in_folder = config_file['in_folder']+dataset+'/input/'+sub+'/t2star/'
    out_folder = config_file['out_folder']+dataset+'/'+sub+'/'
    interm_res_folder = config_file['interm_res_folder']
    brainmask_file = config_file['in_folder']+dataset+'/output/'+sub+'/'+task+'/qBOLD/T1w_coreg/rcBrMsk_CSF.nii'
    file_tag = task+'_acq-fullres_T2star.nii.gz'
    thr = config_file['motion_thr']
    save_magn_nifti = config_file['save_magn_nifti']
    include_transform = config_file['include_transform']
    include_inhomog = config_file['include_inhomog']
    motion_from_pca = config_file['motion_from_pca']
    check_threshold = config_file['check_threshold'] if 'check_threshold' in config_file else False
    if motion_from_pca:
        motion_folder = config_file['motion_folder']
        scenario = config_file['train_test_scenario']
        pca_weight_range = config_file['pca_weight_range'] if 'pca_weight_range' in config_file else 3
    else:
        nr_motion_curve = config_file['motion_curve_nr']
        npy_file = config_file['npy_file']

    if 'motion_from_h5' in config_file:
        motion_from_h5 = config_file['motion_from_h5']
        tag_h5 = config_file['tag_h5']
    else:
        motion_from_h5 = False
        tag_h5 = ''
    if 'motion_level' in config_file:
        motion_level = config_file['motion_level']
        check_motion = True
    else:
        check_motion = False

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print('File: ', in_folder, '**', file_tag)
    print('Results will be saved under: ', out_folder)

    print('############################\nStarting Simulation\n############################')
    if motion_from_h5:
        if isinstance(config_file['simulation_nr'], list):
            simulation_nrs = config_file['simulation_nr']
            motion_files = [glob.glob(out_folder + '**' + tag_h5 + '_' +
                                      str(s) + '**.h5')[0]
                            for s in simulation_nrs]
        else:
            motion_files = sorted(glob.glob(out_folder + '**' + tag_h5 +
                                            '**.h5'))
            simulation_nr = config_file['simulation_nr']
            simulation_nrs = np.arange(simulation_nr,
                                       len(motion_files)+simulation_nr)
    else:
        if isinstance(config_file['simulation_nr'], list):
            simulation_nrs = config_file['simulation_nr']
        else:
            simulation_nrs = [config_file['simulation_nr']]
        motion_files = [0 for s in simulation_nrs]


    dset, affine, header = load_all_echoes(in_folder, file_tag, nr_pe_steps=92)
    brainmask = np.rollaxis(nib.load(brainmask_file).get_fdata(), 2, 0)
    brainmask = brainmask[:, int((np.shape(brainmask)[1]-92)/2):-int((np.shape(brainmask)[1]-92)/2)]

    nr_slices = int(dset.shape[1])
    path_scan_order = interm_res_folder + 'Scan_order_' + str(nr_slices) + '.txt'
    scan_length = int(np.loadtxt(path_scan_order)[-1, 0]) + 1  # duration of a scan in seconds

    for motion_file, simulation_nr in zip(motion_files, simulation_nrs):
        print('Simulating number {}'.format(simulation_nr))
        start = time.time()
        if include_transform:
            if include_inhomog:
                tag = 'sim_b0_rigid_' + str(simulation_nr)
            else:
                tag = 'sim_rigid_' + str(simulation_nr)
        else:
            if include_inhomog:
                tag = 'sim_b0_' + str(simulation_nr)
            else:
                print('Please set either rigid transformations or B0-inhomogeneities (or both) to True.')
                continue
        print('The result is saved with tag: ', tag)

        tag_ = tag
        if check_threshold:
            tag_ = tag_ + '_thr'
            print('Including threshold into simulations.')
        if os.path.exists(out_folder + sub + '_' + task + '_acq-fullres_T2star_' + tag_ + '.h5'):
            print('ERROR: Output file already exits. Please check and maybe change file naming, i.e. simulation_nr.')
            continue

        # import motion data
        # different ways to import the motion data
        if not motion_from_h5:
            if check_motion:
                # only import motion curves where average displacement is within limits defined in motion_level:
                print('Searching motion curves for right motion level:')
                av_magn = -1
                if motion_from_pca:
                    print('Weight range for combining principal components: ', pca_weight_range)
                while not motion_level[1] >= av_magn > motion_level[0]:
                    # import motion data
                    if motion_from_pca:
                        MotionImport = ImportMotionDataPCAfMRI(pc_folder=motion_folder, scenario=scenario,
                                                               scan_length=scan_length, ratio_components=0.2,
                                                               weight_range=pca_weight_range, random_start_time=True, reference_to_0=True)
                    else:
                        MotionImport = ImportMotionDataNpy(npy_file=npy_file, scan_length=scan_length,
                                                           nr_curve=nr_motion_curve, random_start_time=True, reference_to_0=True)
                    times, T, R = MotionImport.get_motion_data(dset.shape)

                    # look at average displacement of a sphere with radius 64mm
                    motion_data = np.array([T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
                    centroids, tr_coords = transform_sphere([12, 35, 92, 112], motion_data,
                                                            pixel_spacing=[3.3, 2, 2], radius=64)
                    # calculate reference through median
                    ind_median_centroid = np.argmin(
                        np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
                    # calculate average voxel displacement magnitude
                    displ = tr_coords - tr_coords[ind_median_centroid]
                    magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
                    av_magn = np.mean(magn)
                    print('\nCurrent average Magnitude:', av_magn, end='')
                print(' accepted')

            else:
                # import a random motion curve (without defining limits)
                if motion_from_pca:
                    print('Weight range for combining prinicpal components: ', pca_weight_range)
                    MotionImport = ImportMotionDataPCAfMRI(pc_folder=motion_folder, scenario=scenario,
                                                           scan_length=scan_length, ratio_components=0.2,
                                                           weight_range=pca_weight_range, random_start_time=True, reference_to_0=True)
                    print('Random PCA-augmented motion curve used for simulation.')
                else:
                    MotionImport = ImportMotionDataNpy(npy_file=npy_file, scan_length=scan_length,
                                                       nr_curve=nr_motion_curve, random_start_time=True, reference_to_0=True)
                times, T, R = MotionImport.get_motion_data(dset.shape)

            # resort the motion parameters to match dimensions of data:
            motion = np.array([times, T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T

        else:
            # import a motion curve from existing simulated data (h5 file)
            motion = h5py.File(motion_file, "r")['Motion_Curve']
            print('Motion scenario: ', motion_file)

        # initialize the simulation:
        Simulation = SimulateMotionForEachReadout(motion, nr_pe_steps=92, brainmask=brainmask, check_threshold=check_threshold,
                                                  path_scan_order=path_scan_order, motion_thr=thr,
                                                  include_transform=include_transform, include_inhomog=include_inhomog)

        magn, mask, full_mask = Simulation.create_mask_from_motion(dset)
        __, soft_mask, __ = Simulation.create_soft_mask_from_motion(dset)

        # Simulate motion:
        dset_sim = Simulation.simulate_all(dset)

        # save the simulated data:
        with h5py.File(out_folder+sub+'_'+task+'_acq-fullres_T2star_'+tag_+'.h5', 'w') as h5_file:
            dset_1 = h5_file.create_dataset(name='Simulated_Data', shape=np.shape(dset_sim), data=dset_sim)
            dset_1.attrs['Original_data_folder'] = in_folder
            dset_1.attrs['Original_data_file_tag'] = file_tag
            dset_2 = h5_file.create_dataset(name='Motion_Curve', shape=np.shape(motion), data=motion)
            dset_2.attrs['Order_of_motion_data'] = 'Times, z-translation, y-translation, x-translation, z-rotation, ' \
                                            'y-rotation, x-rotation'
            dset_3 = h5_file.create_dataset(name='Corruption_Mask', shape=np.shape(mask), data=mask)
            dset_3.attrs['Reference_method'] = 'Median'
            dset_3.attrs['Threshold'] = thr
            dset_3.attrs['Labels_of_mask'] = '0 for motion > threshold, 1 for motion <= threshold (lines that can be included)'
            dset_4 = h5_file.create_dataset(name='Soft_Corruption_Mask', shape=np.shape(soft_mask),
                                                data=soft_mask)
            dset_4.attrs['Note'] = 'Calculated with: soft_mask = 1 - magn_sphere / 2, since voxel size of 2mm'
            dset_5 = h5_file.create_dataset(name='Affine_Nifti_Transform', shape=np.shape(affine), data=affine)
            dset_5.attrs['Note_for_saving'] = 'A) When saving real and imaginary data as nfitis to be further ' \
                                              'processed with Matlab pipeline, remember to add offset=2047 (to get ' \
                                              'the same intensity values as scanner) and save the unscaled nifti ' \
                                              'version with the following commands: 1) extract header from ' \
                                              'original nifti (information in attributes of Simulated data), ' \
                                              '(2) hd = header.set_slope_inter(1, 0), ' \
                                              '(3) sim_nii = nib.Nifti1Image(dset, affine, hd), ' \
                                              '(4) nib.save(sim_nii, path).' \
                                              'B) In general: when saving the niftis for further processing:' \
                                              'remember to perform zerofilling in PE direction up to 112 lines'
            dset_6 = h5_file.create_dataset(name='Brain_mask', shape=np.shape(brainmask), data=brainmask)


        # optionally: save the simulated data as nifti files:
        if save_magn_nifti:
            for i in range(dset_sim.shape[0]):
                hd = header.set_slope_inter(1, 0)
                sim_echo = np.rollaxis(dset_sim[i], 0, 3)
                sim_nii = nib.Nifti1Image(abs(sim_echo), affine, hd)
                nib.save(sim_nii, out_folder+sub+'_'+task+'_acq-fullres_T2star_'+tag + '_magn/echo_' + str(i) + '.nii')

        end = time.time()
        print('############################\n', str(current), ' out of ', str(total), ' done')
        print('############################\nSimulation took: ', (end - start) / 60, ' minutes.'
                                                                                     '\n############################')
        current += 1

