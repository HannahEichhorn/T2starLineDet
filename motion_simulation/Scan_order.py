'''
Script to generate text files containing the acquisition order, which are needed for simulation

Code adapted from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

To run this script: specify input file and output folder
'''

import os
import ismrmrd.xsd
import numpy as np
import matplotlib.pyplot as plt


# files and folders:
filename = "path_to_raw_file.h5"        # raw data file in ISMRMRD format from which the scan order can be extracted
out_folder = "path_to_output_folder"


# Load file
if not os.path.isfile(filename):
    print("%s is not a valid file" % filename)
    raise SystemExit
dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)

header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

# Matrix size
eNx = enc.encodedSpace.matrixSize.x
eNy = enc.encodedSpace.matrixSize.y
eNz = enc.encodedSpace.matrixSize.z
rNx = enc.reconSpace.matrixSize.x
rNy = enc.reconSpace.matrixSize.y
rNz = enc.reconSpace.matrixSize.z

# Field of View
eFOVx = enc.encodedSpace.fieldOfView_mm.x
eFOVy = enc.encodedSpace.fieldOfView_mm.y
eFOVz = enc.encodedSpace.fieldOfView_mm.z
rFOVx = enc.reconSpace.fieldOfView_mm.x
rFOVy = enc.reconSpace.fieldOfView_mm.y
rFOVz = enc.reconSpace.fieldOfView_mm.z

# Number of Slices, Reps, Contrasts, etc.
ncoils = header.acquisitionSystemInformation.receiverChannels
if enc.encodingLimits.slice != None:
    nslices = enc.encodingLimits.slice.maximum + 1
else:
    nslices = 1

if enc.encodingLimits.repetition != None:
    nreps = enc.encodingLimits.repetition.maximum + 1
else:
    nreps = 1

if enc.encodingLimits.contrast != None:
    ncontrasts = enc.encodingLimits.contrast.maximum + 1
else:
    ncontrasts = 1

#  loop through the acquisitions looking for noise scans
firstacq = 0
for acqnum in range(dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)

    # Currently ignoring noise scans
    if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        print("Found noise scan at acq ", acqnum)
        continue
    else:
        firstacq = acqnum
        print("Imaging acquisition starts acq ", acqnum)
        break

# Initialize lists:
rep = []
contrast = []
slice = []
y = []
time = []

# Loop through the rest of the acquisitions and stuff
for acqnum in range(firstacq, dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)

    # Stuff into the buffer
    rep.append(acq.idx.repetition)
    contrast.append(acq.idx.contrast)
    slice.append(acq.idx.slice)
    y.append(acq.idx.kspace_encode_step_1)
    time.append(acq.acquisition_time_stamp)


for i in range(0, 100):
    print(i, contrast[i], slice[i], y[i], time[i])



# add acquisition time defined by TR and echo times:
# average echo times for different slice numbers:
# 30 slices: TR = 1.91056
# 35 slices: TR = 2.22810
# 36 slices: TR = 2.29144
rep_time = 2.29144
echo_time = 0.005001
echo_diff = 0.005
nr_PE = 92
nr_slices = 36
nr_echoes = 12
all_echoes = echo_time + (nr_echoes-1)*echo_diff
gap = (rep_time-all_echoes*nr_slices)/nr_slices

times = []
curr_time = 0
for i in range(0, nr_PE):
    for j in range(0, nr_slices):
        for k in range(0, nr_echoes):
            if k == 0:
                curr_time += echo_time
            else:
                curr_time += echo_diff
            times.append(curr_time)
        curr_time += gap
    curr_time = rep_time * (i+1)

# exclude all data with slice > nr_slices:
rep = np.array(rep)
contrast = np.array(contrast)
y = np.array(y)
slice = np.array(slice)

rep = rep[slice < nr_slices]
contrast = contrast[slice < nr_slices]
y = y[slice < nr_slices]
slice = slice[slice < nr_slices]

save_arr = np.array([times, rep, contrast, slice, y]).T
np.savetxt(out_folder + 'Scan_order_'+str(nr_slices)+'.txt', save_arr,
           header='Timing, repetition, contrast, slice, phase encode step y')


# plot the scan order:
contrast = np.array(contrast)
slice = np.array((slice))
y = np.array(y)
time = np.array(times)


plt.figure(figsize=(15,15))
max_pe = np.amax(y)
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 12)]

# go through contrast by contrast:
for i, color in zip(range(0, 12), colors):
    c = contrast[contrast == i]
    s = slice[contrast == i]
    pe = y[contrast == i]
    t = time[contrast == i]

    pe_ch = pe+s*max_pe
    plt.scatter(t, pe_ch, color=color, label='echo '+str(i), s=2)
plt.legend(loc='best')
plt.xlabel('Acquisition time [s]')
plt.ylabel('Slices and PE lines')

plt.savefig(out_folder + 'Scan_order.png', dpi=300)
plt.show()


# zoom in on only a number of slices and PE lines:
plt.figure()
max_pe = np.amax(y)
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 12)]

# go through contrast by contrast:
for i, color in zip(range(0, 12), colors):
    c = contrast[contrast==i]
    s = slice[contrast == i]
    pe = y[contrast == i]
    t = time[contrast == i]

    c = c[s<7]
    pe = pe[s<7]
    t = t[s<7]
    s = s[s<7]

    c = c[pe > 85]
    t = t[pe > 85]
    s = s[pe > 85]
    pe = pe[pe > 85]

    pe_ch = pe+s*max_pe
    plt.scatter(t, pe_ch, color=color, label='echo '+str(i), s=2)
plt.legend(loc='upper right')
plt.xlabel('Acquisition time [s]')
plt.ylabel('Slices and PE lines')

plt.savefig(out_folder + 'Scan_order_zoomed.png', dpi=300)
plt.show()


# zoom in even more:
plt.figure()
max_pe = np.amax(y)
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 12)]

# go through contrast by contrast:
for i, color in zip(range(0, 12), colors):
    c = contrast[contrast==i]
    s = slice[contrast == i]
    pe = y[contrast == i]
    t = time[contrast == i]

    c = c[s<4]
    pe = pe[s<4]
    t = t[s<4]
    s = s[s<4]

    c = c[pe > 90]
    t = t[pe > 90]
    s = s[pe > 90]
    pe = pe[pe > 90]

    pe_ch = pe+s*max_pe
    plt.scatter(t, pe_ch, color=color, label='echo '+str(i), s=2)
    plt.xlim(0, 0.2)
plt.legend(loc='upper right')
plt.xlabel('Acquisition time [s]')
plt.ylabel('Slices and PE lines')

plt.savefig(out_folder + 'Scan_order_zoomed_more.png', dpi=300)
plt.show()
