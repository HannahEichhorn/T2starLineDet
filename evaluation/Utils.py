"""
Utility functions for evaluating the model's predictions
"""

import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from transforms3d.euler import euler2mat
import medutils


def load_predictions(path, path_sim, thr):
    """ Loading the model's predictions and corresponding targets """
    files = sorted(glob.glob(path))

    preds = []
    targets = []
    filenames = []
    slices = []

    for f in files:
        tmp = np.loadtxt(f, unpack=True)
        # threshold the predictions:
        tmp_thr = np.zeros_like(tmp)
        tmp_thr[tmp > 0.5] = 1
        preds.append(tmp_thr)

        tmp = os.path.basename(f)
        tmp_f = tmp.split('_Slice_')[0]
        filenames.append(tmp_f)
        slices.append(int(tmp.split('_Slice_')[1].split('.txt')[0]))
        # correct h5 file path:
        dataset = tmp_f.split("-sub-")[0]+'/'
        subj = 'sub-'+tmp_f.split("-sub-")[1].split('_task-')[0]+'/'
        with h5py.File(path_sim+dataset+subj+tmp_f.replace(dataset[:-1]+'-', ""), "r") as hf:
            soft_target = hf["Soft_Corruption_Mask"][:]
            soft_target = np.mean(soft_target, axis=0)
        tmp = soft_target[int(tmp.split('_Slice_')[1].split('.txt')[0])]

        # threshold the target masks:
        tmp[tmp < thr] = 0
        tmp[tmp >= thr] = 1
        targets.append(tmp)

    return np.array(preds), np.array(targets), filenames, slices


def Acc(pred, targ, axis=(1, 2)):
    """ Accuracy of prediction compared to target """
    return np.count_nonzero(pred == targ, axis=axis)/np.count_nonzero(targ >= 0, axis=axis)


def ND(pred, targ, axis=(1, 2)):
    """ Rate of not-detected lines """
    return np.count_nonzero((pred == 1)*(targ == 0), axis=axis)/np.count_nonzero(targ >= 0, axis=axis)


def WD(pred, targ, axis=(1, 2)):
    """ Rate of wrongly-detected liens """
    return np.count_nonzero((pred == 0)*(targ == 1), axis=axis)/np.count_nonzero(targ >= 0, axis=axis)


def SingleCoilForwardOp(img, mask, fft_dim=(-2, -1)):
    return medutils.mri.fft2c(img, axes=fft_dim) * mask


def SingleCoilAdjointOp(kspace, mask, fft_dim=(-2, -1)):
    return medutils.mri.ifft2c(kspace * mask, axes=fft_dim)


def quick_imshow(data, indices, title=None, vmin=None, vmax=None):
    if np.iscomplexobj(data):
        data = np.abs(data)
    for i in indices:
        data = data[i]
    plt.imshow(data[:, ::-1].T, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def transform_sphere(dset_shape, motion_parameters, pixel_spacing, radius):
    # get all voxels within sphere around isocenter:
    dim1, dim2, dim3 = dset_shape[-3:]
    zz, xx, yy = np.ogrid[:dim1, :dim2, :dim3]
    zz = zz * pixel_spacing[0]
    xx = xx * pixel_spacing[1]
    yy = yy * pixel_spacing[2]
    center = [np.mean(zz), np.mean(xx), np.mean(yy)]
    d2 = (zz - center[0]) ** 2 + (xx - center[1]) ** 2 + (yy - center[2]) ** 2
    mask = d2 <= radius ** 2
    z, x, y = np.nonzero(mask)
    coords = np.array(list(zip(z, x, y)))
    coords[:, 0] = coords[:, 0] * pixel_spacing[0]
    coords[:, 1] = coords[:, 1] * pixel_spacing[1]
    coords[:, 2] = coords[:, 2] * pixel_spacing[2]

    # reduce number of coordinates to speed up calculation:
    coords = coords[::100]

    # apply the transforms to the coordinates:
    centroids = []
    tr_coords = []
    for pars in motion_parameters:
        T = np.array(pars[0:3]) / np.array(pixel_spacing)
        R = np.array(pars[3:]) * np.pi / 180
        tr_coords_ = np.matmul(coords, euler2mat(*R).T)
        tr_coords_ = tr_coords_ + T
        tr_coords.append(tr_coords_)
        centroids.append(np.mean(tr_coords_, axis=0))

    return np.array(centroids), np.array(tr_coords)


def load_h5_dataset(filename, soft_mask=False, orig_img=False):
    with h5py.File(filename, "r") as hf:
        # load simualated data:
        simulated_image = hf["Simulated_Data"][:]
        target_mask = hf["Corruption_Mask"][:]
        target_mask = np.mean(target_mask, axis=0)
        target_mask[target_mask <= 0.5] = 0
        target_mask[target_mask > 0.5] = 1.0

        if soft_mask:
            soft_target_mask = hf["Soft_Corruption_Mask"][:]
            soft_target_mask = np.mean(soft_target_mask, axis=0)

        if orig_img:
            orig_image = hf["Original_Data"][:]

        original_folder = hf["Simulated_Data"].attrs["Original_data_folder"]
        original_file_tag = hf["Simulated_Data"].attrs["Original_data_file_tag"]

    if not soft_mask:
        if not orig_img:
            return simulated_image, target_mask, original_folder, original_file_tag
        else:
            return simulated_image, target_mask, orig_image, original_folder, original_file_tag
    else:
        if not orig_img:
            return simulated_image, target_mask, soft_target_mask, original_folder, \
                   original_file_tag
        else:
            return simulated_image, target_mask, soft_target_mask, orig_image, original_folder, \
                   original_file_tag


def load_h5_motion_data(filename):
    with h5py.File(filename, "r") as hf:
        # load motion data:
        motion_data = h5py.File(filename, "r")['Motion_Curve']

    centroids, tr_coords = transform_sphere([12, 35, 92, 112], motion_data[:, 1:],
                                            pixel_spacing=[3.3, 2, 2], radius=64)
    # calculate reference through median
    ind_median_centroid = np.argmin(
        np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
    # calculate average voxel displacement magnitude
    displ = tr_coords - tr_coords[ind_median_centroid]
    magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
    av_magn = np.mean(magn)

    return motion_data, av_magn, magn