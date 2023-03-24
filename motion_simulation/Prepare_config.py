"""Script for creating config files for the simulations

To run this script:

(1) Specify input and output folders and files (subject_list, folder_sim_train, folder_sim_test, folder_pc_hpc, config_folder)
(2) Create a csv file containing dataset names, subject names and consensus quality score. 'Example_subject_list.csv' is an example for this.
(3) Adapt the dataset names and tasks below to relevant names and tasks (lines 155-167, 212-220, 256-273, 410-422).
(4) Set train_test_split, create_config_CV and do_CV_split to True.
"""

import os
import numpy as np


def List2String(list):
    tmp = list[0]
    for i in range(1, len(list)):
        tmp += ', ' + list[i]
    return tmp


train_test_split = False                # perform a train test split for the image data
create_config_CV = False                # create configuration files for simulation
do_CV_split = False                     # whether to perform perform 5 train validiation splits in create_config_CV
create_config_CV_all_train = False      # create configuration files for simulation (PCA on all CV data)


# define folders etc.:
# Note: the config files were generated on a different computer than the simulation run, which is why some folders
# need to be defined on both computers
# computer 1:
subject_list = "path_to_file/Example_subject_list.csv"
folder_sim_train = "path_to_simulated_data_CV"
folder_sim_test = "path_to_simulated_data_test/"
config_folder = "output_folder_configs"
# computer 2:
folder_pc_hpc = "path_to_motion_curves_CV/"
folder_in_hpc = "pat_to_input_data"
folder_sim_train_hpc = "path_to_simulated_data_CV"
folder_sim_test_hpc = " path_to_simulated_data_test/"
folder_res_hpc = "pat_to_interm_results"


if train_test_split:
    dataset, subject, score = np.loadtxt(subject_list, skiprows=1, unpack=True, dtype=str, delimiter=',')

    dataset = np.array([d.replace('/', '') for d in dataset])
    score = score.astype(int)

    # sort out all scores >=2
    dataset = dataset[score < 2]
    subject = subject[score < 2]
    score = score[score < 2]

    # create a train test split (subject-wise):
    indices = np.arange(len(subject))
    np.random.shuffle(indices)
    subject = subject[indices]
    dataset = dataset[indices]

    train_subjects, train_datasets = [], []
    test_subjects, test_datasets = [], []
    for ds, sub in zip(dataset, subject):
        if sub in train_subjects or sub in test_subjects:
            continue
        if len(train_subjects) < 92:
            tmp1 = subject[dataset==ds]
            tag = sub.split('-')[1]
            tmp1 = [s for s in tmp1 if tag in s]
            for t in tmp1:
                train_subjects.append(t)
                train_datasets.append(ds)
        else:
            test_subjects.append(sub)
            test_datasets.append(ds)

    np.savetxt(folder_sim_train + "List_subjects_train.txt",
               np.array([train_datasets, train_subjects]).T, fmt='%s')
    np.savetxt(folder_sim_test + "List_subjects_test.txt",
               np.array([test_datasets, test_subjects]).T, fmt='%s')


if create_config_CV:
    if do_CV_split:
        # create config files for simulating cross validation data:
        train_datasets, train_subjects = np.loadtxt(folder_sim_train + "List_subjects_train.txt",
                                                    dtype=str, unpack=True)
        test_datasets, test_subjects = np.loadtxt(folder_sim_test + "List_subjects_test.txt",
                                                  dtype=str, unpack=True)

        # do train validation splits for 5 folds:
        continue_loop = True
        while continue_loop:
            indices = np.arange(len(train_subjects))
            np.random.shuffle(indices)
            subjects = train_subjects[indices]
            datasets = train_datasets[indices]

            val_datasets = {'1':[], '2':[], '3':[], '4':[], '5':[]}
            val_subjects = {'1':[], '2':[], '3':[], '4':[], '5':[]}

            for i in range(1, 6):
                for ds, sub in zip(datasets, subjects):
                    if sub in [l for k in val_subjects.keys() for l in val_subjects[k]]:
                        continue
                    if len(val_subjects[str(i)]) < 18:
                        tmp1 = subjects[datasets == ds]
                        tag = sub.split('-')[1]
                        tmp1 = [s for s in tmp1 if tag in s]
                        for t in tmp1:
                            val_subjects[str(i)].append(t)
                            val_datasets[str(i)].append(ds)

            lengths = [len(val_subjects[str(k)]) for k in range(1, 6)]
            if len(np.unique(lengths)) == 1:
                continue_loop = False

        # save the 5 folds:
        for i in range(1, 6):
            sub_val = val_subjects[str(i)]
            ds_val = val_datasets[str(i)]
            sub_train, ds_train = [], []
            for ds, sub in zip(train_datasets, train_subjects):
                if sub not in sub_val:
                    sub_train.append(sub)
                    ds_train.append(ds)
            print(len(sub_val), len(sub_train))
            np.savetxt(folder_sim_train + "List_subjects_train_Fold_{}.txt".format(i),
                       np.array([ds_train, sub_train]).T, fmt='%s')
            np.savetxt(folder_sim_train + "List_subjects_val_Fold_{}.txt".format(i),
                       np.array([ds_val, sub_val]).T, fmt='%s')


    config_files_train = []
    config_files_val = []
    config_files_test = []
    # train data:
    for i in range(1, 6):
        # for each fold:
        train_datasets, train_subjects = np.loadtxt(folder_sim_train + "List_subjects_train_Fold_{}.txt".format(i),
                                                    dtype=str, unpack=True)

        for simulation in ['b0_rigid', 'rigid']:
            for level, descr, sim_nr in zip([[0, 0.4], [0.4, 0.8], [0.8, 1.2]], ['mild', 'moderate', 'strong'],
                                            [[0, 1], [2, 3], [4, 5]]):
                task = []
                subjects = []
                for ds, sub in zip(train_datasets, train_subjects):
                    tmp = sub.split('-')
                    subj = tmp[0]+'-'+tmp[1]
                    if ds == 'DATA_Bose':
                        task.append('task-eunat')
                        subjects.append(subj)
                    elif ds == 'DATA_Christine':
                        task.append('task-AIR')
                        subjects.append(subj)
                    elif ds == 'DATA_Epp_1_task':
                        task.append('task-control')
                        subjects.append(subj)
                    else:
                        tmp = sub.split('-')
                        task.append('task-' + tmp[2])
                        subjects.append(tmp[0] + '-' + tmp[1])

                if simulation == 'b0_rigid':
                    include_inhom = True
                else:
                    include_inhom = False

                lines = ['dataset: [{}]'.format(List2String(train_datasets)),
                         'subject: [{}]'.format(List2String(subjects)),
                         'task: [{}]'.format(List2String(task)),
                         'in_folder: {}'.format(folder_in_hpc),
                         'out_folder: {}Train_Fold_{}/'.format(folder_sim_train_hpc, i),
                         'interm_res_folder: {}'.format(folder_res_hpc),
                         'motion_folder: PCA_Fold_{}/'.format(folder_pc_hpc, i),
                         'motion_from_pca: True',
                         'train_test_scenario: train',
                         'simulation_nr: {}'.format(sim_nr),
                         'motion_level: {}'.format(level),
                         'motion_thr: 0.5',
                         'include_transform: True',
                         'include_inhomog: {}'.format(include_inhom),
                         'save_magn_nifti: False']

                with open(config_folder + "config_run_all_train_Fold{}_{}_{}.yaml".format(i, simulation, descr),
                          'w') as file:
                    for line in lines:
                        file.write("%s\n" % line)

                config_files_train.append(config_folder + "config_run_all_train_Fold{}_{}_{}.yaml".format(i, simulation,
                                                                                                          descr))


    # val data:
    for i in range(1, 6):
        # for each fold:
        val_datasets, val_subjects = np.loadtxt(folder_sim_train +"List_subjects_val_Fold_{}.txt".format(i),
                                                dtype=str, unpack=True)
        npy_file = folder_pc_hpc + "Motion_Curves_val_Fold_{}.npy".format(i)
        sim_nr = 0
        motion_curve_nr = np.arange(0, len(val_subjects))
        np.random.shuffle(motion_curve_nr)

        for simulation in ['b0_rigid', 'rigid']:
            descr = 0
            for ds, sub, nr in zip(val_datasets, val_subjects, motion_curve_nr):
                if ds == 'DATA_Bose':
                    task = 'task-eunat'
                elif ds == 'DATA_Christine':
                    task = 'task-AIR'
                elif ds == 'DATA_Epp_1_task':
                    task = 'task-control'
                else:
                    tmp = sub.split('-')
                    task = 'task-' + tmp[2]

                if simulation == 'b0_rigid':
                    include_inhom = True
                else:
                    include_inhom = False

                tmp = sub.split('-')
                subj = tmp[0] + '-' + tmp[1]

                lines = ['dataset: [{}]'.format(List2String([ds])),
                         'subject: [{}]'.format(List2String([subj])),
                         'task: [{}]'.format(List2String([task])),
                         'in_folder: {}'.format(folder_in_hpc),
                         'out_folder: {}Val_Fold_{}/'.format(i, folder_sim_train_hpc),
                         'interm_res_folder: {}'.format(folder_res_hpc),
                         'motion_from_pca: False',
                         'npy_file: {}'.format(npy_file),
                         'motion_curve_nr: {}'.format(nr),
                         'simulation_nr: {}'.format(sim_nr),
                         'motion_thr: 0.5',
                         'include_transform: True',
                         'include_inhomog: {}'.format(include_inhom),
                         'save_magn_nifti: False']

                with open(config_folder + "config_run_all_val_Fold{}_{}_{}.yaml".format(i, simulation, descr),
                          'w') as file:
                    for line in lines:
                        file.write("%s\n" % line)

                config_files_val.append(config_folder + "config_run_all_val_Fold{}_{}_{}.yaml".format(i, simulation,
                                                                                                      descr))
                descr += 1


    # test data:
    test_datasets, test_subjects = np.loadtxt(folder_sim_test + "List_subjects_test.txt", dtype=str, unpack=True)
    npy_file = folder_pc_hpc + "Motion_Curves_test.npy"
    sim_nr = 0
    motion_curve_nr = np.arange(0, len(test_subjects))
    np.random.shuffle(motion_curve_nr)

    for simulation in ['b0_rigid', 'rigid']:
        descr = 0
        for ds, sub, nr in zip(test_datasets, test_subjects, motion_curve_nr):
            if ds == 'DATA_Bose':
                task = 'task-eunat'
            elif ds == 'DATA_Christine':
                task = 'task-AIR'
            elif ds == 'DATA_Epp_1_task':
                task = 'task-control'
            else:
                tmp = sub.split('-')
                task = 'task-' + tmp[2]

            if simulation == 'b0_rigid':
                include_inhom = True
            else:
                include_inhom = False

            tmp = sub.split('-')
            subj = tmp[0] + '-' + tmp[1]

            lines = ['dataset: [{}]'.format(List2String([ds])),
                     'subject: [{}]'.format(List2String([subj])),
                     'task: [{}]'.format(List2String([task])),
                     'in_folder: {}'.format(folder_in_hpc),
                     'out_folder: {}'.format(folder_sim_test_hpc),
                     'interm_res_folder: {}'.format(folder_res_hpc),
                     'motion_from_pca: False',
                     'npy_file: {}'.format(npy_file),
                     'motion_curve_nr: {}'.format(nr),
                     'simulation_nr: {}'.format(sim_nr),
                     'motion_thr: 0.5',
                     'include_transform: True',
                     'include_inhomog: {}'.format(include_inhom),
                     'save_magn_nifti: False']

            with open(config_folder + "config_run_all_test_{}_{}.yaml".format(simulation, descr), 'w') as file:
                for line in lines:
                    file.write("%s\n" % line)

            config_files_test.append(config_folder + "config_run_all_test_{}_{}.yaml".format(simulation, descr))
            descr += 1


if create_config_CV_all_train:
    train_datasets, train_subjects = np.loadtxt(folder_sim_train + "List_subjects_train.txt", dtype=str, unpack=True)

    config_files_train = []

    for nr, ind in enumerate([[0, 30], [30, 60], [60, 92]]):
        for simulation in ['b0_rigid', 'rigid']:
            for level, descr, sim_nr in zip([[0, 0.4], [0.4, 0.8], [0.8, 1.2]], ['mild', 'moderate', 'strong'],
                                            [[0, 1], [2, 3], [4, 5]]):
                task = []
                subjects = []
                for ds, sub in zip(train_datasets[ind[0]:ind[1]], train_subjects[ind[0]:ind[1]]):
                    tmp = sub.split('-')
                    subj = tmp[0] + '-' + tmp[1]
                    if ds == 'DATA_Bose':
                        task.append('task-eunat')
                        subjects.append(subj)
                    elif ds == 'DATA_Christine':
                        task.append('task-AIR')
                        subjects.append(subj)
                    elif ds == 'DATA_Epp_1_task':
                        task.append('task-control')
                        subjects.append(subj)
                    else:
                        tmp = sub.split('-')
                        task.append('task-' + tmp[2])
                        subjects.append(tmp[0] + '-' + tmp[1])

                if simulation == 'b0_rigid':
                    include_inhom = True
                else:
                    include_inhom = False

                lines = ['dataset: [{}]'.format(List2String(train_datasets[ind[0]:ind[1]])),
                         'subject: [{}]'.format(List2String(subjects)),
                         'task: [{}]'.format(List2String(task)),
                         'in_folder: {}'.format(folder_in_hpc),
                         'out_folder: {}Train_all/'.format(folder_sim_train_hpc),
                         'interm_res_folder: {}'.format(folder_res_hpc),
                         'motion_folder: {}PCA_all_CV/'.format(folder_pc_hpc),
                         'motion_from_pca: True',
                         'train_test_scenario: train',
                         'simulation_nr: {}'.format(sim_nr),
                         'motion_level: {}'.format(level),
                         'motion_thr: 0.5',
                         'include_transform: True',
                         'include_inhomog: {}'.format(include_inhom),
                         'save_magn_nifti: False']

                with open(config_folder + "config_run_all_train_all_{}_{}_{}.yaml".format(simulation, descr, nr),
                          'w') as file:
                    for line in lines:
                        file.write("%s\n" % line)

                config_files_train.append(config_folder + "config_run_all_train_all_{}_{}_{}.yaml".format(simulation, descr, nr))
