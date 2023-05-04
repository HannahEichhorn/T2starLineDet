"""
Script to create symbolic links to data sources (to enable easier handling
when training on different machines)

Note: when running on different dataset, remember to exchange the dataset names.
"""
import os.path
import subprocess
import glob


def CreateSymbLink(file_name, link_name):
    subprocess.run('ln -s ' + file_name + ' ' + link_name, shell=True)
    return 0


do_HPC_CV = False
do_test = True
do_WS_CV = True


# input and output folders:
# machine 1:
folder_in_hpc = 'input_folder_machine_1'
folder_in_hpc_test = 'input_folder_test_machine_1'
folder_out_hpc = 'output_folder_machine_1'
# machine 2:
folder_in_WS = 'input_folder_machine_2'
folder_in_WS_test = 'input_folder_test_machine_2'
folder_out_WS = "output_folder_machine_2"


dataset_names = ['DATA_Bose', 'DATA_Christine', 'DATA_Epp_4_task',
                 'DATA_Epp_2_task', 'DATA_Epp_1_task', 'DATA_Hechler']


if do_HPC_CV:
    # Run on HPC (machine 1)
    folds = ['Fold_'+ i for i in ['1']]
    tag = ['b0_rigid']
    descr = tag[0]+'**'+tag[1] if len(tag) > 1 else tag[0]

    for fold in folds:
        # train:
        all_files = glob.glob(folder_in_hpc+'Train_'+fold+'/**/**/**sim_'+descr+'**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for training: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_hpc+'train_T2star_'+descr.replace('**', '_')+'_'+fold+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)


        # validation:
        all_files = glob.glob(folder_in_hpc + 'Val_' + fold + '/**/**/**sim_' + descr + '**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for validation: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_hpc+'val_T2star_'+descr.replace('**', '_')+'_'+fold+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)


    # test:
    if do_test:
        all_files = glob.glob(folder_in_hpc_test + '**/**/**sim_' + descr + '**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for test: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_hpc+'test_T2star_'+descr.replace('**', '_')+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)



if do_WS_CV:
    # run on WS (machine 2)
    folds = ['Fold_'+ i for i in ['1']]
    tag = ['b0_rigid']
    descr = tag[0]+'**'+tag[1] if len(tag) > 1 else tag[0]

    for fold in folds:
        # train:
        all_files = glob.glob(folder_in_WS+'Train_'+fold+'/**/**/**sim_'+descr+'**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for training: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_WS+'train_T2star_'+descr.replace('**', '_')+'_'+fold+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)


        # validation:
        all_files = glob.glob(folder_in_WS + 'Val_' + fold + '/**/**/**sim_' + descr + '**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for validation: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_WS+'val_T2star_'+descr.replace('**', '_')+'_'+fold+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)


    # test:
    if do_test:
        all_files = glob.glob(folder_in_WS_test + '**/**/**sim_' + descr + '**')
        if len(tag) == 1:
            all_files = [f for f in all_files if '_thr' not in f]
        print('#####################')
        print('Number of files for test: ', len(all_files))
        print('#####################')

        for file_name in all_files:
            print(file_name)
            folder = folder_out_WS+'test_T2star_'+descr.replace('**', '_')+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            for d in dataset_names:
                if d in file_name:
                    ds = d
            link_name = folder + ds + '-' + os.path.basename(file_name)
            CreateSymbLink(file_name, link_name)
