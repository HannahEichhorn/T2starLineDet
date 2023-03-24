"""
Utility functions for motion simulation
"""

from transforms3d.affines import decompose, compose
from transforms3d.euler import mat2euler, euler2mat
import numpy as np
from scipy.ndimage import rotate, shift
import glob
import nibabel as nib
from datetime import datetime
from skimage.morphology import erosion, dilation


def load_all_echoes(in_folder, descr, nr_pe_steps=92, nr_echoes=12, offset=2047):
    """Load all echoes for one acquisition into one array as complex dataset

    :param in_folder: input folder
    :param descr: string describing the files to look for
    :param nr_pe_steps: number of phase encoding steps to be simulated.
    :param nr_echoes: number of echoes that are expected. The default is 12.
    :param offset: offset of scanner for saved intensity values. Default for Philips is 2047.
    :return: a complex array of shape [nr_echoes, nr_slices, n_y, n_x]
    :return: an array of shape [4, 4] containing the affine transform for saving as nifti. For saving, please
             remember to apply np.rollaxis of the images to get the shape [n_y, n_x, n_slices]
    """

    files_real = sorted(glob.glob(in_folder+'*real*'+descr))
    files_imag = sorted(glob.glob(in_folder + '*imaginary*' + descr))

    if len(files_real) != nr_echoes:
        print('ERROR: Less than 12 real images found.')
        print(files_real)

    if len(files_imag) != nr_echoes:
        print('ERROR: Less than 12 imaginary images found.')
        print(files_imag)

    shape = np.shape(np.rollaxis(nib.load(files_real[0]).get_fdata(), 2, 0))
    dataset = np.zeros((nr_echoes, shape[0], nr_pe_steps, shape[2]), dtype=complex)

    for re, im, i in zip(files_real, files_imag, range(0, nr_echoes)):
        # load the unscaled nifti (intensities need to match exactly with what scanner saved)
        # substract offset of 2047 since scanner (Philips) shifted the intensities
        tmp = (np.rollaxis(nib.load(files_real[i]).dataobj.get_unscaled(), 2, 0) - offset) + \
              1j * (np.rollaxis(nib.load(files_imag[i]).dataobj.get_unscaled(), 2, 0) - offset)
        dataset[i] = tmp[:, int((np.shape(tmp)[1]-nr_pe_steps)/2):-int((np.shape(tmp)[1]-nr_pe_steps)/2)]

    affine = nib.load((files_real[0])).affine
    header = nib.load(files_real[0]).header

    return dataset, affine, header


def apply_transform_image(image, par, pixel_spacing=[3.3, 2, 2]):
    """Apply translation and rotation to the input image.

    :param image: image in the shape [n_sl, n_y, n_x]
    :param par: list or array of three translation and three rotation parameters in mm and degrees;
                ordered in slice_dir, y, x
    :param pixel_spacing: list of pixel spacing for converting mm to pixel; ordered in slice_dir, y, x
    :return: transformed image
    """

    T = np.array(par[0:3]) / np.array(pixel_spacing)  # convert mm to pixel

    transf_image = image
    for angle, axes in zip(par[3:], [(1,2), (0,2), (0,1)]):
        transf_image = rotate(transf_image, angle, axes, reshape=False)

    transf_image = shift(transf_image, T)

    return transf_image


def ifft2c(x, shape=None, dim=(-2,-1)):
    """Centered Inverse Fourier transform"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho', s=shape), axes=dim)


def fft2c(x, shape=None, dim=(-2, -1)):
    """Centered Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho', s=shape), axes=dim)


def transform_sphere(dset_shape, motion_parameters, pixel_spacing, radius):
    """Rigidly transform a sphere with given motion parameters"""

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


def transf_from_parameters(T, R):
    """
    Use python module transforms3d to extract transformation matrix from
    translation and rotation parameters.

    Parameters
    ----------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.
    Returns
    -------
    A : numpy array (4x4)
        transformation matrix.
    """
    R_mat = euler2mat(R[2] * np.pi / 180, R[1] * np.pi / 180, R[0] * np.pi / 180)
    A = compose(T, R_mat, np.ones(3))

    return A


def parameters_from_transf(A):
    '''
    Use python module transforms3d to extract translation and rotation
    parameters from transformation matrix.
    Parameters
    ----------
    A : numpy array (4x4)
        transformation matrix.
    Returns
    -------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.
    '''
    T, R_, Z_, S_ = decompose(A)
    al, be, ga = mat2euler(R_)
    R = np.array([ga * 180 / np.pi, be * 180 / np.pi, al * 180 / np.pi])

    return np.array(T), R


class SimulateMotionForEachReadout:
    def __init__(self, motion_tracking, nr_pe_steps, brainmask, check_threshold=False,
                 include_transform=True, include_inhomog=False,
                 path_scan_order="path_to_scan_order/Scan_order.txt",
                 pixel_spacing=[3.3, 2, 2], echo0=0.005001, echo_diff=0.005, motion_thr=1):
        super().__init__()

        self.motion_parameters = motion_tracking[:,1:]
        self.motion_times = motion_tracking[:,0]
        self.pixel_spacing = pixel_spacing
        temp = np.loadtxt(path_scan_order, unpack=True)
        self.acq_times, self.reps, self.echoes, self.slices, self.ys = temp
        self.nr_pe_steps = nr_pe_steps
        self.include_inhomog = include_inhomog
        self.include_transform = include_transform
        self.echo0 = echo0
        self.echo_diff = echo_diff
        self.motion_thr = motion_thr
        self.brainmask = brainmask
        self.check_threshold = check_threshold


    def create_mask_from_motion(self, dataset,  radius=64):
        """
        Calculate a mask depending on the average displacement of a sphere that
        is transformed according to the motion curve

        The mask is 1, if the displacement is smaller/equal than the threshold
        (self.motion_thr), and 0 otherwise.
        """
        centroids, tr_coords = transform_sphere(dataset.shape, self.motion_parameters,
                                                self.pixel_spacing, radius)

        # calculate reference through median
        ind_median_centroid = np.argmin(np.sqrt(np.sum((centroids-np.median(centroids, axis=0))**2, axis=1)))

        # calculate average voxel displacement magnitude
        displ = tr_coords - tr_coords[ind_median_centroid]
        magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
        av_magn = np.mean(magn, axis=1)


        # 0 for motion > threshold, 1 for motion <= threshold (lines that can be included)
        motion_class = np.zeros(len(self.motion_times), dtype=int)
        motion_class[av_magn <= self.motion_thr] = 1
        print(
            '{}% of all time points <= {}mm displacement'.format(np.round(len(av_magn[av_magn <= self.motion_thr]) / len(av_magn) * 100, 3),
                                                                 self.motion_thr))

        max_slices = dataset.shape[1]
        mask = np.ones_like(dataset)

        not_acquired = np.amax(dataset.shape[-2]) - self.nr_pe_steps
        ys_sh = self.ys + int(not_acquired / 2)

        for t, r, e, s, y in zip(self.acq_times, self.reps, self.echoes, self.slices, ys_sh):
            if s < max_slices:
                idx = np.argmin(np.abs(self.motion_times-t))
                mask[int(e), int(s), int(y)] = motion_class[idx]
        mask = mask.astype(int)
        red_mask = mask[:, :, :, 0]    # only pick one value per pe line

        return av_magn, red_mask, mask


    def create_soft_mask_from_motion(self, dataset,  radius=64):
        """
        Calculate a "soft mask" depending on the average displacement of a
        sphere that is transformed according to the motion curve

        The mask is 1, if the displacement is 0 mm, it is 0 if the displacement
        is 2 mm and decreases linearly in between.
        """
        centroids, tr_coords = transform_sphere(dataset.shape, self.motion_parameters,
                                                self.pixel_spacing, radius)

        # calculate reference through median
        ind_median_centroid = np.argmin(np.sqrt(np.sum((centroids-np.median(centroids, axis=0))**2, axis=1)))

        # calculate average voxel displacement magnitude
        displ = tr_coords - tr_coords[ind_median_centroid]
        magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
        av_magn = np.mean(magn, axis=1)

        # create soft mask:
        soft_class = 1 - av_magn / 2
        soft_class[soft_class < 0] = 0

        max_slices = dataset.shape[1]
        mask = np.ones_like(dataset)

        not_acquired = np.amax(dataset.shape[-2]) - self.nr_pe_steps
        ys_sh = self.ys + int(not_acquired / 2)

        for t, r, e, s, y in zip(self.acq_times, self.reps, self.echoes, self.slices, ys_sh):
            if s < max_slices:
                idx = np.argmin(np.abs(self.motion_times-t))
                mask[int(e), int(s), int(y)] = soft_class[idx]
        red_mask = mask[:, :, :, 0]    # only pick one value per pe line

        return av_magn, red_mask, mask


    def simulate_one_readout(self, image, parameters, apply_transform=True, w_inhom=False, TE=0):
        """
        Simulate motion for one readout line

        :param image: image corresponding to that readout line (one echo only!)
        :param parameters: motion parameters corresponding to readout timing
        :param w_inhom: map of magnetic field inhomogeneities
        :param TE: echo time
        :return: kspace of transformed image

        Note: in theory, the multiplication with B0 inhomogeneities is to be
        performed after rotating and translating the image. Here, we perform it
        before, since we multiply with randomly generated B0 inhomogeneities
        anyway. Like this, the brain mask does not need to be rigidly
        transformed together with the image.
        """

        if not isinstance(w_inhom, bool):
            image_tr = image*np.exp(-2j*np.pi*w_inhom*TE)
        else:
            image_tr = image

        if apply_transform:
            image_tr = apply_transform_image(image_tr, parameters, self.pixel_spacing)

        return fft2c(image_tr)


    def create_random_B0_inhom(self, dataset, times):
        """
        Create random B0 inhomogeneity maps with maximal values sampled from np.random.uniform(2, 5) [Hz]
        """

        phase_dset = np.angle(dataset) + np.pi
        mask_ = self.brainmask

        cross = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])

        mask = np.zeros_like(mask_)
        for i in range(len(mask_)):
            tmp = dilation(mask_[i], cross)
            tmp = dilation(tmp, cross)
            tmp = dilation(tmp, cross)
            tmp = erosion(tmp, cross)
            tmp = erosion(tmp, cross)
            tmp = erosion(tmp, cross)
            mask[i] = tmp

        mask = np.tile(mask, (12, 1, 1, 1))
        phase_dset = phase_dset * mask

        sim_field_map = np.zeros((len(times), *phase_dset[0].shape))

        for t in range(1, int(times[-1]/10)+1):
            # simulate varying magnetic field for each slice:
            order = 3
            coefficients = np.random.random((order + 1) ** 2)

            half_shape = np.array(phase_dset[0,0].shape) / 2
            ranges = [np.arange(-n, n) + 0.5 for n in half_shape]
            x_mesh, y_mesh = np.asarray(np.meshgrid(*ranges, indexing='ij'))

            intens_grad = np.zeros_like(x_mesh)
            i = 0
            for x_order in range(order + 1):
                for y_order in range(order + 1 - x_order):
                    coefficient = coefficients[i]
                    intens_grad += (
                            coefficient
                            * x_mesh ** x_order
                            * y_mesh ** y_order
                    )
                    i += 1

            # random flip:
            flip = np.random.randint(0, 2)
            if flip:
                intens_grad = intens_grad[::-1]
            flip = np.random.randint(0, 2)
            if flip:
                intens_grad = intens_grad[:, ::-1]

            # get it for all slices:
            intens_grad = np.tile(intens_grad, (phase_dset[0].shape[0], 1, 1))

            # define maximum range of field differences to be between 2 and 5 Hz:
            tmp = intens_grad * mask[0]
            max_int = np.amax(tmp)
            intens_grad /= max_int
            intens_grad *= np.random.uniform(2, 5)

            for ind, ti in enumerate(times):
                if ti < t*10:
                    if ti > (t-1)*10:
                        sim_field_map[ind] = intens_grad

        return sim_field_map


    def simulate_all(self, dataset, radius_sph=64):
        """Combine the simulations for all readout lines"""
        max_slices = dataset.shape[1]
        if max_slices < np.amax(self.slices):
            print('Scan order does not contain enough slices!!!')
        kspace_tr = fft2c(dataset)

        # calculate random B0 inhomogeneities
        inhomogeneities = self.create_random_B0_inhom(dataset, self.motion_times)

        # look at motion magnitude
        centroids, tr_coords = transform_sphere(dataset.shape, self.motion_parameters,
                                                self.pixel_spacing, radius=radius_sph)
        # calculate reference through median
        ind_median_centroid = np.argmin(np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
        # calculate average voxel displacement magnitude
        displ = tr_coords - tr_coords[ind_median_centroid]
        magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
        av_magn = np.mean(magn, axis=1)

        # initialize parameters
        count = 0
        last_idx = np.ones(dataset.shape[0])*-1
        last_pars = np.ones((dataset.shape[0], 6))*-1
        last_kspace = np.zeros_like(dataset)

        # only apply simulation to central pe lines:
        not_acquired = np.amax(kspace_tr.shape[-2]) - self.nr_pe_steps
        ys_sh = self.ys + int(not_acquired / 2)

        for t, r, e, s, y in zip(self.acq_times, self.reps, self.echoes, self.slices, ys_sh):
            if s < max_slices:
                idx = np.argmin(np.abs(self.motion_times-t))
                pars = self.motion_parameters[idx]

                if np.array_equal(pars, last_pars[int(e)]):
                    # if no new motion parameters, the transformed kspace does not need to be recalculated!
                    kspace_tr[int(e), int(s), int(y)] = last_kspace[int(e), int(s), int(y)]

                else:
                    if not self.include_inhomog:
                        w_inhom = False
                        TE = 0
                    else:
                        # don't add inhomogeneities for negligible motion
                        if av_magn[idx] < self.motion_thr:
                            w_inhom = False
                            TE = 0
                        else:
                            w_inhom = inhomogeneities[idx]
                            TE = self.echo0 + int(e)*self.echo_diff

                    if self.check_threshold:
                        # only simulate for non-negligible motion
                        if av_magn[idx] > self.motion_thr:
                            kspace = self.simulate_one_readout(dataset[int(e)], pars,
                                                               apply_transform=self.include_transform,
                                                               w_inhom=w_inhom, TE=TE)
                            kspace_tr[int(e), int(s), int(y)] = kspace[int(s), int(y)]
                            last_kspace[int(e)] = kspace
                            last_idx[int(e)] = idx
                            last_pars[int(e)] = pars
                    else:

                        kspace = self.simulate_one_readout(dataset[int(e)], pars,
                                                           apply_transform=self.include_transform,
                                                           w_inhom=w_inhom, TE=TE)
                        kspace_tr[int(e), int(s), int(y)] = kspace[int(s), int(y)]
                        last_kspace[int(e)] = kspace
                        last_idx[int(e)] = idx
                        last_pars[int(e)] = pars

                count += 1
                if count % 5000 == 0:
                    print(count, ' done')

        return ifft2c(kspace_tr)


class ImportMotionDataPCAfMRI:
    """Importing PCA-augmented fMRI motion data.

    fMRI motion data was decomposed with sklearn.decomposition.PCA and the
    first principal components (unit eigenvectors) can now be combined (weighted
    by corresponding eigenvalues) with the mean motion curves to generate new
    motion curves.

    Parameters
    ----------
    pc_folder : str
        path to folder containing the results of the PC analysis. The folder
        should contain two text files (expl_var_<scenario>.txt and
        Mean_<scenario>.txt)) as well as a subfolder components_<scenario>,
        which contains text files with the unit eigenvectors
        (pc_00.txt, pc_o1.txt, ..).
    scenario : str
        string indicating which scenario (i.e. train or test data) should be
        loaded.
    scan_length : int
        defining the necessary length of the motion curve.
    ratio_components : float
        defining which percentage of the principal components should be used
        for generating new motion curves. The default is 0.2.
    weight_range : int or float
        this parameter defines how much the principal components are varied
        when combining them with the mean (+- weight_range*sqrt(eigenvalue)).
        The default is 3.
    random_start_time : bool
        whether a random time in the motion curve should be picked as start
        time if the curve is longer than scan_length
    reference_to_0 : bool
        whether the motion curve should be transformed so that the median
        position (reference) is at the zero-point.

    Atributes
    -------
    get_motion_data : numpy array
        outputs time points, translational and rotational parameters which are
        prepared as stated above.
    """

    def __init__(self, pc_folder, scenario, scan_length, ratio_components=0.2,
                 weight_range=3, random_start_time=True, reference_to_0=True):
        super().__init__()
        self.reference_to_0 = reference_to_0

        # load the necessary data from PCA:
        mean_data = np.loadtxt(pc_folder + 'Mean_' + scenario + '.txt')[:, 1:].T
        self.seconds = np.loadtxt(pc_folder + 'Mean_' + scenario + '.txt')[:, 0]
        components_files = sorted(glob.glob(pc_folder + 'pc**.txt'))
        total_components = int(ratio_components * len(components_files))
        expl_var = np.loadtxt(pc_folder + 'expl_var_' + scenario +
                              '.txt')[:total_components, 1]
        components = np.zeros((total_components, *mean_data.shape))
        for i in range(0, total_components):
            components[i] = np.loadtxt(components_files[i])[:, 1:].T

        # combine mean and principal components:
        weight = np.random.uniform(-weight_range, weight_range, expl_var.shape)
        variation = np.rollaxis(np.repeat([weight * np.sqrt(expl_var)],
                                          components.shape[1], axis=0), 0, 2)
        variation = np.rollaxis(np.repeat([variation], components.shape[2],
                                          axis=0), 0, 3)
        variation = variation * components
        variation = np.sum(variation, axis=0)
        self.motion_data = mean_data + variation

        # only take time points within scan duration, sample randomly throughout time:
        if random_start_time:
            last_time = self.seconds[-1] - scan_length
            if last_time < 1:
                random_start = 0
                print("ERROR in ImportMotionDataPCAfMRI: the loaded motion data "
                      "is shorter than the needed scan length!")
            else:
                random_start = np.random.randint(0, last_time)
            ind = np.intersect1d(np.where(self.seconds >= random_start),
                                 np.where(self.seconds <= (random_start + scan_length)))
            self.motion_data = self.motion_data[:, ind]
            self.seconds = self.seconds[ind]
            self.seconds = self.seconds - self.seconds[0]
        else:
            self.motion_data = self.motion_data[:, self.seconds < scan_length]
            self.seconds = self.seconds[self.seconds < scan_length]


    def get_motion_data(self, dset_shape, pixel_spacing=[3.3, 2, 2], radius=64):
        """Get motion parameters.

        If self.reference_to_0 is set to True,
        the transformation parameters are first transformed so that the
        reference (median) position corresponds to a zero transformation.

        Parameters
        ----------
        dset_shape : numpy array
            shape of the dataset.
        pixel_spacing : list of length 3
            spacing of the voxels in z, x and y-direction
        radius : int or float
            radius of the sphere used for calculating reference position
        Returns
        -------
        seconds : numpy array
            time points in seconds
        T : numpy array
            translation parameters.
        R : numpy array
            rotation angles in degrees.
        """

        # get motion parameters (PCA output is already in mm and degree)
        T, R = self.motion_data[:3].T, self.motion_data[3:].T

        if not self.reference_to_0:
            return self.seconds, T, R

        else:
            # calculate reference through median of sphere's centroids
            tmp = np.array([T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
            centroids, _ = transform_sphere(dset_shape, tmp, pixel_spacing, radius)

            ind_median_centroid = np.argmin(np.sqrt(np.sum(
                (centroids - np.median(centroids, axis=0)) ** 2, axis=1)))

            # transform all matrices so that ind_median_centroid corresponds to identity:
            matrices = np.zeros((len(T), 4, 4))
            for i in range(len(T)):
                matrices[i] = transf_from_parameters(T[i], R[i])

            tr_matrices = np.matmul(np.linalg.inv(matrices[ind_median_centroid]),
                                    matrices)

            # get motion parameters
            T_reference0, R_reference0 = np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for i in range(len(T)):
                T_reference0[i], R_reference0[i] = parameters_from_transf(tr_matrices[i])

            return self.seconds, T_reference0, R_reference0


class ImportMotionDataNpy:
    """Importing PCA-augmented fMRI motion data (Old version).

    fMRI motion data was decomposed with sklearn.decomposition.PCA and the
    first principal components (unit eigenvectors) can now be combined (weighted
    by corresponding eigenvalues) with the mean motion curves to generate new
    motion curves.

    Parameters
    ----------
    npy_file : str
        path to folder containing the results of the PC analysis. The folder
        should contain two text files (expl_var_<scenario>.txt and
        Mean_<scenario>.txt)) as well as a subfolder components_<scenario>,
        which contains text files with the unit eigenvectors
        (pc_00.txt, pc_o1.txt, ..).
    scan_length : int
        defining the necessary length of the motion curve.
    nr_curve : int
        number of the curve in the npy file
    random_start_time : bool
        whether a random time in the motion curve should be picked as start
        time if the curve is longer than scan_length
    reference_to_0 : bool
        whether the motion curve should be transformed so that the median
        position (reference) is at the zero-point.

    Atributes
    -------
    get_motion_data : numpy array
        outputs time points, translational and rotational parameters which are
        prepared as stated above.
    """

    def __init__(self, npy_file, scan_length, nr_curve,
                 random_start_time=True, reference_to_0=True):
        super().__init__()
        self.reference_to_0 = reference_to_0

        # load the relevant file and extract the data:
        npy_data = np.load(npy_file, allow_pickle=True)[()]
        self.seconds = npy_data['Time_seconds']
        self.motion_data = np.zeros((6, len(self.seconds)))
        for i, t in enumerate(['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']):
            self.motion_data[i] = npy_data[t][nr_curve]

        # only take time points within scan duration, sample randomly throughout time:
        if random_start_time:
            last_time = self.seconds[-1] - scan_length
            if last_time < 1:
                random_start = 0
                print("ERROR in ImportMotionDataPCAfMRI: the loaded motion data "
                      "is shorter than the needed scan length!")
            else:
                random_start = np.random.randint(0, last_time)
            ind = np.intersect1d(np.where(self.seconds >= random_start),
                                 np.where(self.seconds <= (random_start + scan_length)))
            self.motion_data = self.motion_data[:, ind]
            self.seconds = self.seconds[ind]
            self.seconds = self.seconds - self.seconds[0]
        else:
            self.motion_data = self.motion_data[:, self.seconds < scan_length]
            self.seconds = self.seconds[self.seconds < scan_length]


    def get_motion_data(self, dset_shape, pixel_spacing=[3.3, 2, 2], radius=64):
        """Get motion parameters.

        If self.reference_to_0 is set to True,
        the transformation parameters are first transformed so that the
        reference (median) position corresponds to a zero transformation.

        Parameters
        ----------
        dset_shape : numpy array
            shape of the dataset.
        pixel_spacing : list of length 3
            spacing of the voxels in z, x and y-direction
        radius : int or float
            radius of the sphere used for calculating reference position
        Returns
        -------
        seconds : numpy array
            time points in seconds
        T : numpy array
            translation parameters.
        R : numpy array
            rotation angles in degrees.
        """

        # get motion parameters (PCA output is already in mm and degree)
        T, R = self.motion_data[:3].T, self.motion_data[3:].T

        if not self.reference_to_0:
            return self.seconds, T, R

        else:
            # calculate reference through median of sphere's centroids
            tmp = np.array([T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
            centroids, _ = transform_sphere(dset_shape, tmp, pixel_spacing, radius)

            ind_median_centroid = np.argmin(np.sqrt(np.sum(
                (centroids - np.median(centroids, axis=0)) ** 2, axis=1)))

            # transform all matrices so that ind_median_centroid corresponds to identity:
            matrices = np.zeros((len(T), 4, 4))
            for i in range(len(T)):
                matrices[i] = transf_from_parameters(T[i], R[i])

            tr_matrices = np.matmul(np.linalg.inv(matrices[ind_median_centroid]),
                                    matrices)

            # get motion parameters
            T_reference0, R_reference0 = np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for i in range(len(T)):
                T_reference0[i], R_reference0[i] = parameters_from_transf(tr_matrices[i])

            return self.seconds, T_reference0, R_reference0
