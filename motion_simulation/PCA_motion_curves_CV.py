"""Script for performing cross validation splits and principal component
analysis on train sets

To run this script:

(1) Specify input and output folders (folder_orig and folder_pc)
(2) Set new_train_val_split, perform_new_pca and perform_pca_all_cv_data to True.
"""

import glob
import os.path
import numpy as np
from sklearn.decomposition import PCA
from Utils import parameters_from_transf, transf_from_parameters
import scipy
import random
random.seed(0)
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ff506e', '#69005f', 'tab:gray', '#0065bd', 'tab:olive', 'peru'])
from Utils import transform_sphere, ImportMotionDataPCAfMRI

new_train_val_split = False         # split up the data into test and 5 folds of train and validation sets
perform_new_pca = False             # perform principal component analysis on the 5 train splits
perform_pca_all_cv_data = False     # perform principal component analysis on merged train and validation sets

# folders etc.:
folder_orig = "path_to_motion_data"
folder_pc = "path_to_pca"


if new_train_val_split:
    '''1) Load all files and bring them into the same format:'''
    files = glob.glob(folder_orig+'**.txt')

    all_curves = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': []}

    dsc_curves = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': [], 'filename': []}
    fMRI_curves = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': [], 'filename': []}
    CVR_curves = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': [], 'filename': []}
    CVR_curves_450 = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': [], 'filename': []}
    CVR_curves_750 = {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [], 'r_z': [], 'filename': []}
    TRs = {'_dsc': 1.3, '_fMRI': 1.5, '_CVR': 1.2}

    for filename in files:
        # load motion data
        motion_data = np.loadtxt(filename)
        for a, b in zip(['_dsc', '-fMRI', '_CVR'], [dsc_curves, fMRI_curves, CVR_curves]):
            if a in filename:
                T, R = motion_data[:, :3], motion_data[:, 3:] * 180 / np.pi  # convert radians to degrees
                for t, i in zip(['t_x', 't_y', 't_z'], [0, 1, 2]):
                    b[t].append(T[:, i])
                for r, i in zip(['r_x', 'r_y', 'r_z'], [0, 1, 2]):
                    b[r].append(R[:, i])
                b['filename'].append(filename)

    # resort CVR curves depending on length:
    for t in CVR_curves:
        if t != 'filename':
            for nr, x in enumerate(CVR_curves[t]):
                if len(x) == 450:
                    CVR_curves_450[t].append(x)
                    if t == 't_x':
                        CVR_curves_450['filename'].append(CVR_curves['filename'][nr])
                elif len(x) == 750:
                    CVR_curves_750[t].append(x)
                    if t == 't_x':
                        CVR_curves_750['filename'].append(CVR_curves['filename'][nr])
                else:
                    print('Length does not match: ', len(x))

    # convert into arrays:
    for t in dsc_curves:
        dsc_curves[t] = np.array(dsc_curves[t])
    for t in fMRI_curves:
        fMRI_curves[t] = np.array(fMRI_curves[t])
    for t in CVR_curves_450:
        CVR_curves_450[t] = np.array(CVR_curves_450[t])
    for t in CVR_curves_750:
        CVR_curves_750[t] = np.array(CVR_curves_750[t])

    print('#####################')
    print('Number of DSC: ', len(dsc_curves['t_x']))
    print('Number of fMRI: ', len(fMRI_curves['t_x']))
    print('Number of CVR: ', len(CVR_curves_450['t_x']), '(450) + ', len(CVR_curves_750['t_x']), '(750)')
    print('#####################')


    # get equal length for all curves:
    # (1) combine two dsc curves
    # (2) split all other curves into same length
    length_all = 150
    for t in dsc_curves:
        if t not in ['filename']:
            tmp = dsc_curves[t]
            tmp_l = []

            for a, b in zip(tmp[::2], tmp[1::2]):
                tmp_l.append(np.concatenate((a, b)))

            seconds = np.arange(0, len(tmp_l[0]) * TRs['_dsc'], TRs['_dsc'])
            seconds_all = np.arange(0, len(tmp_l[0]) * TRs['_dsc'], TRs['_fMRI'])

            tmp_itp = scipy.interpolate.interp1d(seconds, tmp_l)
            tmp_itp = tmp_itp(seconds_all)

            for t_intpl in tmp_itp:
                all_curves[t].append(t_intpl[0:length_all])

    for t in fMRI_curves:
        if t not in ['filename']:
            tmp = fMRI_curves[t]

            # does not have to be interpolated
            length = len(tmp[0])

            for i in range(0, int(length / length_all)):
                for a in tmp:
                    all_curves[t].append(a[i * 150:150 + i * 150])


    for curves in [CVR_curves_450, CVR_curves_750]:
        for t in curves:
            if t not in ['filename']:
                tmp = curves[t]

                # interpolate:
                seconds = np.arange(0, len(tmp[0]) * TRs['_CVR'], TRs['_CVR'])
                seconds_all = np.arange(0, len(tmp[0]) * TRs['_CVR'], TRs['_fMRI'])

                tmp_itp = scipy.interpolate.interp1d(seconds, tmp)
                tmp_itp = tmp_itp(seconds_all)

                length = len(tmp_itp[0])

                for i in range(0, int(length / length_all)):
                    for a in tmp_itp:
                        all_curves[t].append(a[i * 150:150 + i * 150])


    # correct all curves where first point is not identity:
    # transform with inverse of first point
    for i in range(0, len(all_curves['t_x'])):
        if all_curves['t_x'][i][0] != 0:
            T = np.concatenate(([all_curves['t_x'][i]],
                                [all_curves['t_y'][i]],
                                [all_curves['t_z'][i]]), axis=0).T
            R = np.concatenate(([all_curves['r_x'][i]],
                                [all_curves['r_y'][i]],
                                [all_curves['r_z'][i]]), axis=0).T

            matrices = np.zeros((len(T), 4, 4))
            for j in range(len(T)):
                matrices[j] = transf_from_parameters(T[j], R[j])

            tr_matrices = np.matmul(np.linalg.inv(matrices[0]),
                                    matrices)

            # get motion parameters
            T_0, R_0 = np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for j in range(len(T)):
                T_0[j], R_0[j] = parameters_from_transf(tr_matrices[j])

            all_curves['t_x'][i] = T_0[:, 0]
            all_curves['t_y'][i] = T_0[:, 1]
            all_curves['t_z'][i] = T_0[:, 2]
            all_curves['r_x'][i] = R_0[:, 0]
            all_curves['r_y'][i] = R_0[:, 1]
            all_curves['r_z'][i] = R_0[:, 2]


    ''' 2) calculate motion statistics for all_curves '''
    RMS = []
    motion_free = []
    max_motion = []

    for i in range(0, len(all_curves['t_x'])):
        # look at average displacement of a sphere with radius 64mm
        motion_data = np.array([all_curves['t_z'][i],
                                all_curves['t_y'][i],
                                all_curves['t_x'][i],
                                all_curves['r_z'][i],
                                all_curves['r_y'][i],
                                all_curves['r_x'][i]]).T
        centroids, tr_coords = transform_sphere([12, 35, 92, 112], motion_data,
                                                pixel_spacing=[3.3, 2, 2], radius=64)
        # calculate reference through median
        ind_median_centroid = np.argmin(
            np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
        # calculate average voxel displacement magnitude
        displ = tr_coords - tr_coords[ind_median_centroid]
        magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
        RMS.append(np.mean(magn))
        magn_sph = np.mean(magn, axis=1)
        motion_free.append(np.count_nonzero(magn_sph < 0.5)/len(magn_sph))
        max_motion.append(np.amax(magn_sph))


    ''' 3) Divide the data in train and test sets and try to have as close means of the motion metrics as possible:'''
    RMS = np.array(RMS)
    motion_free = np.array(motion_free)
    max_motion = np.array(max_motion)

    best_indices = []
    best_diff_RMS, best_diff_motion_free, best_diff_max_motion = 100, 100, 100
    best_diff_RMS_max, best_diff_motion_free_max, best_diff_max_motion_max = 100, 100, 100
    for i in range(0, 15000):
        perm_ind = np.random.permutation(len(RMS))
        curr_RMS, curr_motion_free, curr_max_motion = RMS[perm_ind], motion_free[perm_ind], max_motion[perm_ind]
        diff_mean_RMS = abs(np.mean(curr_RMS[0:108]) - np.mean(curr_RMS[108:]))
        diff_mean_motion_free = abs(np.mean(curr_motion_free[0:108]) - np.mean(curr_motion_free[108:]))
        diff_mean_max_motion = abs(np.mean(curr_max_motion[0:108]) - np.mean(curr_max_motion[108:]))
        diff_max_RMS = abs(np.max(curr_RMS[0:108]) - np.max(curr_RMS[108:]))
        diff_max_motion_free = abs(np.min(curr_motion_free[0:108]) - np.min(curr_motion_free[108:]))
        diff_max_max_motion = abs(np.max(curr_max_motion[0:108]) - np.max(curr_max_motion[108:]))
        if diff_mean_RMS < best_diff_RMS and diff_mean_motion_free < best_diff_motion_free and diff_mean_max_motion < best_diff_max_motion:
            if diff_max_RMS < best_diff_RMS_max and diff_max_motion_free < best_diff_motion_free_max and diff_max_max_motion < best_diff_max_motion_max:
                print(str(i)+' Better split found')
                print(diff_mean_RMS, diff_mean_motion_free, diff_mean_max_motion)
                best_indices = perm_ind
                best_diff_RMS = diff_mean_RMS
                best_diff_motion_free = diff_mean_motion_free
                best_diff_max_motion = diff_mean_max_motion

    # divide up after best indices:
    RMS_train, RMS_test = RMS[best_indices][0:108], RMS[best_indices][108:]
    motion_free_train, motion_free_test = motion_free[best_indices][0:108], motion_free[best_indices][108:]
    max_motion_train, max_motion_test = max_motion[best_indices][0:108], max_motion[best_indices][108:]

    all_curves_train, all_curves_test = {}, {}

    for t in all_curves:
        all_curves_train[t] = np.array(all_curves[t])[best_indices][0:108]
        all_curves_test[t] = np.array(all_curves[t])[best_indices][108:]


    print("Train data: RMS {}, motion free ratio {}, maximum displacement {}".format(np.mean(RMS_train),
                                                                                     np.mean(motion_free_train),
                                                                                     np.mean(max_motion_train)))
    print("Test data: RMS {}, motion free ratio {}, maximum displacement {}".format(np.mean(RMS_test),
                                                                                     np.mean(motion_free_test),
                                                                                     np.mean(max_motion_test)))

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.scatter(np.zeros_like(RMS_train), RMS_train, label='Train')
    plt.scatter(np.ones_like(RMS_test), RMS_test, label='Test')
    plt.plot([0], [np.mean(RMS_train)], '_', c='tab:gray', ms=20)
    plt.plot([1], [np.mean(RMS_test)], '_', c='tab:gray', ms=20)
    plt.legend()
    plt.xticks([])
    plt.ylabel("RMS displacement sphere")
    plt.subplot(1, 3, 2)
    plt.scatter(np.zeros_like(motion_free_train), np.array(motion_free_train)*100, label='Train')
    plt.scatter(np.ones_like(motion_free_test), np.array(motion_free_test)*100, label='Test')
    plt.plot([0], [np.mean(motion_free_train)*100], '_', c='tab:gray', ms=20)
    plt.plot([1], [np.mean(motion_free_test)*100], '_', c='tab:gray', ms=20)
    plt.legend()
    plt.xticks([])
    plt.ylabel("Motion Free Percentage")
    plt.subplot(1, 3, 3)
    plt.scatter(np.zeros_like(max_motion_train), max_motion_train, label='Train')
    plt.scatter(np.ones_like(max_motion_test), max_motion_test, label='Test')
    plt.plot([0], [np.mean(max_motion_train)], '_', c='tab:gray', ms=20)
    plt.plot([1], [np.mean(max_motion_test)], '_', c='tab:gray', ms=20)
    plt.legend()
    plt.xticks([])
    plt.ylabel("maximum displacement sphere")
    plt.tight_layout()
    plt.show()


    ''' 3) Divide the train data in train and val folds:'''
    best_indices = []
    best_diff_RMS, best_diff_motion_free, best_diff_max_motion = 100, 100, 100
    best_diff_RMS_max, best_diff_motion_free_max, best_diff_max_motion_max = 100, 100, 100
    for i in range(0, 25000):
        perm_ind = np.random.permutation(len(RMS_train))
        curr_RMS, curr_motion_free, curr_max_motion = RMS_train[perm_ind], motion_free_train[perm_ind], max_motion_train[perm_ind]

        diff_mean_RMS = np.std([np.mean(curr_RMS[0:20]), np.mean(curr_RMS[20:40]),
                                np.mean(curr_RMS[40:60]), np.mean(curr_RMS[60:80]),
                                np.mean(curr_RMS[80:100])])
        diff_mean_motion_free = np.std([np.mean(curr_motion_free[0:20]), np.mean(curr_motion_free[20:40]),
                                np.mean(curr_motion_free[40:60]), np.mean(curr_motion_free[60:80]),
                                np.mean(curr_motion_free[80:100])])
        diff_mean_max_motion = np.std([np.mean(curr_max_motion[0:20]), np.mean(curr_max_motion[20:40]),
                                np.mean(curr_max_motion[40:60]), np.mean(curr_max_motion[60:80]),
                                np.mean(curr_max_motion[80:100])])
        diff_max_RMS = np.std([np.max(curr_RMS[0:20]), np.max(curr_RMS[20:40]),
                                np.max(curr_RMS[40:60]), np.max(curr_RMS[60:80]),
                                np.max(curr_RMS[80:100])])
        diff_max_motion_free = np.std([np.min(curr_motion_free[0:20]), np.min(curr_motion_free[20:40]),
                               np.min(curr_motion_free[40:60]), np.min(curr_motion_free[60:80]),
                               np.min(curr_motion_free[80:100])])
        diff_max_max_motion = np.std([np.max(curr_max_motion[0:20]), np.max(curr_max_motion[20:40]),
                               np.max(curr_max_motion[40:60]), np.max(curr_max_motion[60:80]),
                               np.max(curr_max_motion[80:100])])

        if diff_mean_RMS < best_diff_RMS and diff_mean_motion_free < best_diff_motion_free and diff_mean_max_motion < best_diff_max_motion:
            if diff_max_RMS < best_diff_RMS_max and diff_max_motion_free < best_diff_motion_free_max and diff_max_max_motion < best_diff_max_motion_max:
                print(str(i) + ' Better split found')
                print(diff_mean_RMS, diff_mean_motion_free, diff_mean_max_motion)
                best_indices = perm_ind
                best_diff_RMS = diff_mean_RMS
                best_diff_motion_free = diff_mean_motion_free
                best_diff_max_motion = diff_mean_max_motion

    # divide up after best indices:
    splits_val = [best_indices[0:20], best_indices[20:40], best_indices[40:60], best_indices[60:80], best_indices[80:100]]
    RMS_CV_val, motion_free_CV_val, max_motion_CV_val = [], [], []
    RMS_CV_train, motion_free_CV_train, max_motion_CV_train = [], [], []
    for split in splits_val:
        split_train = np.array([ind for ind in best_indices if ind not in split])
        RMS_CV_val.append(RMS_train[split])
        RMS_CV_train.append(RMS_train[split_train])
        motion_free_CV_val.append(motion_free_train[split])
        motion_free_CV_train.append(motion_free_train[split_train])
        max_motion_CV_val.append(max_motion_train[split])
        max_motion_CV_train.append(max_motion_train[split_train])

    all_curves_CV_train, all_curves_CV_val = [{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}]
    for dict_train, dict_val, split in zip(all_curves_CV_train, all_curves_CV_val, splits_val):
        for t in all_curves_train:
            split_train = np.array([ind for ind in best_indices if ind not in split])
            dict_train[t] = np.array(all_curves_train[t])[split_train]
            dict_val[t] = np.array(all_curves_train[t])[split]


    print("CV Train data: RMS {}, motion free ratio {}, maximum displacement {}".format(np.mean(np.array(RMS_CV_train), axis=1),
                                                                                     np.mean(np.array(motion_free_CV_train), axis=1),
                                                                                     np.mean(np.array(max_motion_CV_train), axis=1)))
    print("CV Val data: RMS {}, motion free ratio {}, maximum displacement {}".format(np.mean(np.array(RMS_CV_val), axis=1),
                                                                                     np.mean(np.array(motion_free_CV_val), axis=1),
                                                                                     np.mean(np.array(max_motion_CV_val), axis=1)))

    ''' Save the motion curves for train and test'''
    # test data:
    all_curves_test['Time_seconds'] = np.arange(0, all_curves_test['t_x'].shape[1] * 1.5, 1.5)
    np.save(folder_pc + 'Motion_Curves_test.npy', all_curves_test)
    # can be loaded with: test = np.load(folder_pc+'temp.npy', allow_pickle=True)[()]

    # CV splits:
    for i in range(0, 5):
        all_curves_CV_train[i]['Time_seconds'] = np.arange(0, all_curves_CV_train[i]['t_x'].shape[1] * 1.5, 1.5)
        all_curves_CV_val[i]['Time_seconds'] = np.arange(0, all_curves_CV_val[i]['t_x'].shape[1] * 1.5, 1.5)
        np.save(folder_pc + 'Motion_Curves_train_Fold_{}.npy'.format(str(i+1)), all_curves_CV_train[i])
        np.save(folder_pc + 'Motion_Curves_val_Fold_{}.npy'.format(str(i + 1)), all_curves_CV_val[i])


if perform_new_pca:
    for i in range(0, 5):
        all_curves_train = np.load(folder_pc+'Motion_Curves_train_Fold_{}.npy'.format(str(i+1)), allow_pickle=True)[()]
        seconds_all = all_curves_train['Time_seconds']
        folder_out = folder_pc+'PCA_Fold_'+str(i+1)+'/'
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        # calculate mean curve and subtract them from the individual curves::
        mean_train = np.zeros((7, len(all_curves_train['t_x'][0])))
        for i, t in enumerate(['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']):
            mean_train[i + 1] = np.mean(all_curves_train[t], axis=0)
            all_curves_train[t] -= np.mean(all_curves_train[t], axis=0)

        plt.plot(seconds_all, mean_train[1])
        plt.show()

        mean_train[0] = seconds_all

        np.savetxt(folder_out + 'Mean_train.txt', mean_train.T,
                   header='time, t_x, t_y, t_z, r_x, r_y, r_z in s, mm and deg')

        # stack all axes and perform PCA together:
        stacked_curves = all_curves_train['t_x']
        for t in ['t_y', 't_z', 'r_x', 'r_y', 'r_z']:
            stacked_curves = np.concatenate((stacked_curves, all_curves_train[t]), axis=1)
        pca = PCA()
        pca.fit(stacked_curves.T)
        pca_train_tmp = pca.transform(stacked_curves.T).T
        expl_var_train = pca.explained_variance_
        expl_var_rat_train = pca.explained_variance_ratio_

        pca_train = np.zeros((len(all_curves_train['t_x']), 7, len(all_curves_train['t_x'][0])))
        length = len(all_curves_train['t_x'][0])
        for i in range(0, 6):
            pca_train[:, i + 1, :] = pca_train_tmp[:, length * i:length * (i + 1)]

        # normalize the principal components:
        magn = np.rollaxis(np.repeat([np.sqrt(np.sum(pca_train ** 2, axis=2))], length, axis=0), 0, 3)
        pca_train = pca_train / (magn+1e-8)

        pca_train[:, 0] = seconds_all

        for i in range(0, len(pca_train)):
            if i < 10:
                num = '0'+str(i)
            else:
                num = str(i)
            np.savetxt(folder_out + 'pc_'+num+'.txt', pca_train[i].T,
                       header='time, t_x, t_y, t_z, r_x, r_y, r_z in s, mm and deg')

        expl_var_train = np.concatenate(([np.arange(0, len(pca_train))], [expl_var_train]), axis=0).T
        np.savetxt(folder_out + 'expl_var_train.txt', expl_var_train,
                   header='principal component, explained variance for t_x, t_y, t_z, '
                          'r_x, r_y, r_z')
        expl_var_rat_train = np.concatenate(([np.arange(0, len(pca_train))], [expl_var_rat_train]), axis=0).T
        np.savetxt(folder_out + 'expl_var_ratio_train.txt', expl_var_rat_train,
                   header='principal component, explained variance ratio for t_x, t_y, '
                          't_z, r_x, r_y, r_z')


if perform_pca_all_cv_data:
    all_curves_train = np.load(folder_pc + 'Motion_Curves_train_Fold_1.npy', allow_pickle=True)[()]
    all_curves_val = np.load(folder_pc + 'Motion_Curves_val_Fold_1.npy', allow_pickle=True)[()]
    seconds_all = all_curves_train['Time_seconds']
    folder_out = folder_pc + 'PCA_all_CV/'
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # combine train and validation curves:
    all_curves = all_curves_train
    for t in ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']:
        for val in all_curves_val[t]:
            all_curves[t] = np.append(all_curves[t], [val], axis=0)


    # calculate mean curve and subtract them from the individual curves::
    mean_train = np.zeros((7, len(all_curves['t_x'][0])))
    for i, t in enumerate(['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']):
        mean_train[i + 1] = np.mean(all_curves[t], axis=0)
        all_curves[t] -= np.mean(all_curves[t], axis=0)

    plt.plot(seconds_all, mean_train[1])
    plt.show()

    mean_train[0] = seconds_all

    np.savetxt(folder_out + 'Mean_train.txt', mean_train.T,
               header='time, t_x, t_y, t_z, r_x, r_y, r_z in s, mm and deg')

    # stack all axes and perform PCA together:
    stacked_curves = all_curves['t_x']
    for t in ['t_y', 't_z', 'r_x', 'r_y', 'r_z']:
        stacked_curves = np.concatenate((stacked_curves, all_curves[t]), axis=1)
    pca = PCA()
    pca.fit(stacked_curves.T)
    pca_train_tmp = pca.transform(stacked_curves.T).T
    expl_var_train = pca.explained_variance_
    expl_var_rat_train = pca.explained_variance_ratio_

    pca_train = np.zeros((len(all_curves['t_x']), 7, len(all_curves['t_x'][0])))
    length = len(all_curves['t_x'][0])
    for i in range(0, 6):
        pca_train[:, i + 1, :] = pca_train_tmp[:, length * i:length * (i + 1)]

    # normalize the principal components:
    magn = np.rollaxis(np.repeat([np.sqrt(np.sum(pca_train ** 2, axis=2))], length, axis=0), 0, 3)
    pca_train = pca_train / (magn + 1e-8)

    pca_train[:, 0] = seconds_all

    for i in range(0, len(pca_train)):
        if i < 100:
            num = '0' + str(i)
        else:
            num = str(i)
        if i < 10:
            num = '00' + str(i)

        np.savetxt(folder_out + 'pc_' + num + '.txt', pca_train[i].T,
                   header='time, t_x, t_y, t_z, r_x, r_y, r_z in s, mm and deg')

    expl_var_train = np.concatenate(([np.arange(0, len(pca_train))], [expl_var_train]), axis=0).T
    np.savetxt(folder_out + 'expl_var_train.txt', expl_var_train,
               header='principal component, explained variance for t_x, t_y, t_z, '
                      'r_x, r_y, r_z')
    expl_var_rat_train = np.concatenate(([np.arange(0, len(pca_train))], [expl_var_rat_train]), axis=0).T
    np.savetxt(folder_out + 'expl_var_ratio_train.txt', expl_var_rat_train,
               header='principal component, explained variance ratio for t_x, t_y, '
                      't_z, r_x, r_y, r_z')
