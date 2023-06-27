"""
Script for evaluating the model's performance and performing exemplary weighted
TV reconstructions

To run this script:
(1) add input and output folder paths (line 16 to 18)
(2) insert the correct weights folder names (lines 24, 25, 111, 292)
(3) adapt filenames of example data (lines 112, 113, 294, 295)
"""

import medutils
from medutils.optimization import TVOptimizer
from Utils import *
from motion_simulation.Utils import load_all_echoes
import matplotlib.pyplot as plt
import os
import numpy as np


# input and output folders / files:
folder_in = "path_to_predictions_folder/"
folder_sim = "path_to_test_data/"
folder_out = "path_to_output_folder/"
folder_in_real = "path_to_real_data"


""" 1) Plot simulation examples for different echo times: """
slice_nr = 15
thr_sim = 0.75
thr_mm = 0.5

files = [folder_sim + "DATA_Hechler/sub-p024/sub-p024_task-pe_acq-fullres_T2star_sim_b0_rigid_0.h5",
         folder_sim + "DATA_Hechler/sub-p024/sub-p024_task-pe_acq-fullres_T2star_sim_rigid_0.h5"]
datasets = ["DATA_Hechler", "DATA_Hechler"]

orig_images, sim_images = [], []
for coil_nr in [2, 11]:
    for filename, dataset in zip(files, datasets):
        # load simulated data:
        simulated_image, target_mask, soft_target_mask, original_image, original_folder, original_file_tag = load_h5_dataset(
            filename,
            soft_mask=True,
            orig_img=True)

        sim_image = simulated_image[coil_nr, slice_nr]
        soft_mask = soft_target_mask[slice_nr]
        orig_image = original_image[coil_nr, slice_nr]
        original_kspace = medutils.mri.fft2c(orig_image)

        kspace = medutils.mri.fft2c(sim_image)
        cond = np.rollaxis(np.tile(soft_mask, (112, 1)), 0, 2)
        kspace[cond >= thr_sim] = original_kspace[cond >= thr_sim]

        # Lazy normalization to [0, 255]
        norm = np.max(np.abs(orig_image)) / 255.
        kspace /= norm
        orig_image /= norm
        sim_image = medutils.mri.ifft2c(kspace)

        orig_images.append(orig_image)
        sim_images.append(sim_image)

for echo, i in zip([2,11], [0,2]):
    quick_imshow(orig_images[i], [], 'Original echo {}'.format(echo+1), vmin=0, vmax=255)

for echo, i in zip([2,11], [0,2]):
    quick_imshow(sim_images[i], [], 'B0 echo {}'.format(echo+1), vmin=0, vmax=255)

for echo, i in zip([2,11], [1,3]):
    quick_imshow(sim_images[i], [], 'rigid echo {}'.format(echo+1), vmin=0, vmax=255)

dset, affine, header = load_all_echoes(folder_in_real, 'task-rand_acq-fullres_T2star.nii.gz', nr_pe_steps=92)

quick_imshow(dset[2, 20], [], 'real echo {}'.format(echo + 1))
quick_imshow(dset[11, 20], [], 'real echo {}'.format(echo+1))


""" 2) Performance metrics: """
# WandB ID of relevant trainings
ids_b0_rigid = ["2023_03_04_15_54_08_597069", "2023_03_04_15_54_50_567495",
                "2023_03_04_15_55_28_304315", "2023_03_04_15_56_01_451835"]

sim_thrs = [0.25, 0.5, 0.75, 1.0]
label_thrs = [0.875, 0.75, 0.625, 0.5]

# load the predictions and targets:
predictions_b0, targets_b0 = [], []
for id_b0, thr in zip(ids_b0_rigid, label_thrs):
    preds_b0, targ_b0, filenames_b0, slices_b0 = load_predictions(folder_in + id_b0 + "/Predictions/test/**.txt",
                                                                  folder_sim, thr)
    predictions_b0.append(preds_b0)
    targets_b0.append(targ_b0)

predictions_b0 = np.array(predictions_b0)
targets_b0 = np.array(targets_b0)


# calculate accuracy and rates of ND and WD lines
accs_b0, NDs_b0, WDs_b0 = [], [], []
acc_b0, ND_b0, WD_b0 = [], [], []
acc_b0_std, ND_b0_std, WD_b0_std = [], [], []
for i in range(0, 4):
    accs, NDs, WDs = [], [], []
    for j in range(0, 492):
        accs.append(Acc(predictions_b0[i, j], targets_b0[i, j], axis=None)*100)
        NDs.append(ND(predictions_b0[i, j], targets_b0[i, j], axis=None)*100)
        WDs.append(WD(predictions_b0[i, j], targets_b0[i, j], axis=None)*100)
    accs_b0.append(accs)
    NDs_b0.append(NDs)
    WDs_b0.append(WDs)
    acc_b0.append(np.mean(accs))
    acc_b0_std.append(np.std(accs))
    ND_b0.append(np.mean(NDs))
    ND_b0_std.append(np.std(NDs))
    WD_b0.append(np.mean(WDs))
    WD_b0_std.append(np.std(WDs))


# plot the performance metrics
plt.figure(figsize=(14, 3))
plt.subplot(1, 3, 1)
plt.violinplot(accs_b0, positions=sim_thrs, showmeans=True, showextrema=False, widths=0.2)
plt.xlabel("Threshold for motion simulation $d_{min}$ [mm]", fontsize=12)
plt.ylabel("Accuracy [%]", fontsize=12)
plt.xticks([0.25, 0.5, 0.75, 1.0])
plt.text(-0.13, 107, '(a)', fontsize=14, weight='bold')
for i in range(0,4):
    plt.text(sim_thrs[i], 104, str(np.round(acc_b0[i], 2))+' %', ha='center')
plt.ylim(0, 115)
plt.subplot(1, 3, 2)
plt.violinplot(NDs_b0, positions=sim_thrs, showmeans=True, showextrema=False, widths=0.2)
plt.xlabel("Threshold for motion simulation $d_{min}$ [mm]", fontsize=12)
plt.ylabel("Rate of ND lines [%]", fontsize=12)
plt.xticks([0.25, 0.5, 0.75, 1.0])
for i in range(0,4):
    plt.text(sim_thrs[i], 13, str(np.round(ND_b0[i], 2))+' %', ha='center')
plt.ylim(-0.5, 15)
plt.text(-0.13, 15/115*107, '(b)', fontsize=14, weight='bold')
plt.subplot(1, 3, 3)
plt.violinplot(WDs_b0, positions=sim_thrs, showmeans=True, showextrema=False, widths=0.2)
plt.xlabel("Threshold for motion simulation $d_{min}$ [mm]", fontsize=12)
plt.ylabel("Rate of WD lines [%]", fontsize=12)
plt.xticks([0.25, 0.5, 0.75, 1.0])
plt.text(-0.13, 110/115*107, '(c)', fontsize=14, weight='bold')
heights = [102, 40, 25, 22]
for i in range(0, 4):
    plt.text(sim_thrs[i], heights[i], str(np.round(WD_b0[i], 2))+' %', ha='center')
plt.ylim(-2, 110)
plt.subplots_adjust(wspace=0.3)
plt.savefig(folder_out + "Performance_diff_thr.pdf", format='pdf', bbox_inches="tight")
plt.show()

print('Mean Accuracies for simulations with B0:')
print(acc_b0)
print(acc_b0_std)
print('Mean ND rates for simulations with B0:')
print(ND_b0)
print(ND_b0_std)
print('Mean WD rates for simulations with B0:')
print(WD_b0)
print(WD_b0_std)


""" 3) Example TV reconstructions: """
coil_nr = 11
slice_nr = 15
thr_sim = 0.75
thr_mm = 0.5
id_05 = "2023_03_04_15_54_50_567495"

files = [folder_sim + "DATA_Hechler/sub-p024/sub-p024_task-pe_acq-fullres_T2star_sim_b0_rigid_0.h5"]
datasets = ["DATA_Hechler"]

motion_datasets, displacements = [], []
for filename, dataset in zip(files, datasets):
    print("Reconstructions for: ", filename)
    # load simulated data:
    simulated_image, target_mask, soft_target_mask, original_image, original_folder, original_file_tag = load_h5_dataset(
        filename,
        soft_mask=True,
        orig_img=True)

    motion_data, av_magn, magn = load_h5_motion_data(filename)
    print('Average displacement during scan: {}'.format(av_magn))
    motion_datasets.append(motion_data)
    displacements.append(magn)

    sim_image = simulated_image[coil_nr, slice_nr]
    soft_mask = soft_target_mask[slice_nr]
    orig_image = original_image[coil_nr, slice_nr]
    original_kspace = medutils.mri.fft2c(orig_image)

    mask = np.zeros_like(soft_mask)
    mask[soft_mask >= thr_sim] = 1

    # load the predicted mask:
    pred_file = "{}{}/Predictions/test/{}-{}_Slice_{}.txt".format(folder_in,
                                                                  id_05,
                                                                  dataset,
                                                                  os.path.basename(filename),
                                                                  slice_nr)

    # load and threshold the predictions:
    tmp = np.loadtxt(pred_file, unpack=True)
    mask_pred = np.zeros_like(tmp)
    mask_pred[tmp > 0.5] = 1
    print('Accuracy of mask: ', np.count_nonzero(mask_pred == mask) / np.count_nonzero(mask >= 0))


    kspace = medutils.mri.fft2c(sim_image)
    cond = np.rollaxis(np.tile(soft_mask, (112, 1)), 0, 2)
    kspace[cond >= thr_sim] = original_kspace[cond >= thr_sim]

    # Lazy normalization to [0, 255]
    norm = np.max(np.abs(orig_image)) / 255.
    kspace /= norm
    orig_image /= norm
    sim_image = medutils.mri.ifft2c(kspace)


    def prox_sc_weighted_mri(x, y, alpha):
        """ Proximal operator to weighted DC term for target mask """
        s = medutils.mri.fft2c(y)
        mask_ = mask.copy()
        mask_[mask_ == 0] = 0.25    # weighted DC
        fraction = 1 / (1 + alpha * mask_.reshape((-1, 1)) ** 2)

        return medutils.mri.ifft2c(fraction * (medutils.mri.fft2c(x) + alpha * mask_.reshape((-1, 1)) ** 2 * s))


    def prox_sc_weighted_pred_mri(x, y, alpha):
        """ Proximal operator to weighted DC term for predicted mask """
        s = medutils.mri.fft2c(y)
        mask_ = mask_pred.copy()
        mask_[mask_ == 0] = 0.25  # weighted DC
        fraction = 1 / (1 + alpha * mask_.reshape((-1, 1)) ** 2)

        return medutils.mri.ifft2c(fraction * (medutils.mri.fft2c(x) + alpha * mask_.reshape((-1, 1)) ** 2 * s))


    # with target mask:
    optimizer_weighted = TVOptimizer(mode='2d', lambd=2, prox_h=prox_sc_weighted_mri)
    img_weighted = optimizer_weighted.solve(sim_image, 1000)

    # with predicted mask:
    optimizer_weighted_pred = TVOptimizer(mode='2d', lambd=2, prox_h=prox_sc_weighted_pred_mri)
    img_weighted_pred = optimizer_weighted_pred.solve(sim_image, 1000)

    print('Statistics:')
    print('PSNR motion-corrupted', medutils.measures.psnr(sim_image, orig_image))
    print('PSNR weighted TV', medutils.measures.psnr(img_weighted, orig_image))
    print('PSNR weighted pred TV', medutils.measures.psnr(img_weighted_pred, orig_image))
    print(' ')
    print('SSIM motion-corrupted', medutils.measures.ssim(sim_image, orig_image))
    print('SSIM weighted TV', medutils.measures.ssim(img_weighted, orig_image))
    print('SSIM weighted pred TV', medutils.measures.ssim(img_weighted_pred, orig_image))

    quick_imshow(orig_image, [], 'Ground truth', vmin=0, vmax=255)
    quick_imshow(sim_image, [], 'Motion corrupted', vmin=0, vmax=255)
    quick_imshow(img_weighted, [], 'Weighted TV Recon', vmin=0, vmax=255)
    quick_imshow(img_weighted_pred, [], 'Weighted Pred TV Recon', vmin=0, vmax=255)


""" 3) Supplementary material: """
# plot the motion data corresponding to above reconstructions:
plt.figure(figsize=(10, 4))
titles = ['']

nr = 0
motion_data = motion_datasets[0]
displ = displacements[0]
displ = np.mean(displ, axis=1)
displ[displ < thr_mm] = 0
displ[displ >= thr_mm] = 1

ax1 = plt.subplot(2, 1, 1)
plt.title(titles[nr], fontsize=20)
legend = True
for x_, t in zip(motion_data[:, 0], displ):
    if t == 1:
        if legend:
            plt.axvspan(x_-0.75, x_+0.75, facecolor='lightgrey', label='d > 0.5mm')
            legend = False
        else:
            plt.axvspan(x_ - 0.75, x_ + 0.75, facecolor='lightgrey')
plt.plot(motion_data[:, 0], motion_data[:, 1], label='T_z')
plt.plot(motion_data[:, 0], motion_data[:, 2], label='T_y')
plt.plot(motion_data[:, 0], motion_data[:, 3], label='T_x')
plt.ylabel('Translation [mm]', fontsize=11)
handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=8, ncol=2)
plt.xticks(color='w')
ax2 = plt.subplot(2, 1, 2)
legend = True
for x_, t in zip(motion_data[:, 0], displ):
    if t == 1:
        if legend:
            plt.axvspan(x_ - 0.75, x_ + 0.75, facecolor='lightgrey', label='d > 0.5mm')
            legend = False
        else:
            plt.axvspan(x_ - 0.75, x_ + 0.75, facecolor='lightgrey')
plt.plot(motion_data[:, 0], motion_data[:, 4], label='R_z')
plt.plot(motion_data[:, 0], motion_data[:, 5], label='R_y')
plt.plot(motion_data[:, 0], motion_data[:, 6], label='R_x')
handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=8, ncol=2)
plt.ylabel('Rotation [°]', fontsize=11)
plt.xlabel('Time [s]', fontsize=11)
plt.savefig(folder_out + "Suppl_Motion_Curves.pdf", format='pdf')
plt.show()


# Example motion curve for illustrating motion simulation:
plt.figure(figsize=(10, 3))
for motion_data in [motion_datasets[0]]:
    plt.subplot(2, 1, 1)
    plt.plot(motion_data[:, 0], motion_data[:, 1], label='T_z')
    plt.plot(motion_data[:, 0], motion_data[:, 2], label='T_y')
    plt.plot(motion_data[:, 0], motion_data[:, 3], label='T_x')
    plt.ylabel('Translation [mm]', fontsize=11)
    plt.legend(loc='upper left', fontsize=8)
    plt.xticks(color='w')
    plt.subplot(2, 1, 2)
    plt.plot(motion_data[:, 0], motion_data[:, 4], label='R_z')
    plt.plot(motion_data[:, 0], motion_data[:, 5], label='R_y')
    plt.plot(motion_data[:, 0], motion_data[:, 6], label='R_x')
    plt.legend(loc='upper left', fontsize=8)
    plt.ylabel('Rotation [°]', fontsize=11)
    plt.xlabel('Time [s]', fontsize=11)
plt.savefig(folder_out + "Illustr_Sim_Motion_Curve.png", format='png', bbox_inches='tight')
plt.show()
