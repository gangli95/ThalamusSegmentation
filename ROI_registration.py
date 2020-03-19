import nibabel as nib
import numpy as np
import os
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imaffine import AffineRegistration, AffineMap, MutualInformationMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.align.metrics import CCMetric
from dipy.align.transforms import  RigidTransform3D, AffineTransform3D
from savemat import saveAffineMat, loadAffineMat
from dipy.viz import regtools


def ROI_registration(datapath, template, t1, b0, roi):
    
    t1_path = datapath + '/' + t1
    b0_path = datapath + '/' + b0
    roi_path = datapath + '/' + roi
    template_path = datapath + '/' + template
    
    template_img, template_affine = load_nifti(template_path)
    t1_img, t1_affine = load_nifti(t1_path)
    b0_img, b0_affine = load_nifti(b0_path)
    roi_img, roi_affine = load_nifti(roi_path)

    #diff2struct affine registartion

    moving = b0_img
    moving_grid2world = b0_affine
    static = t1_img
    static_grid2world = t1_affine
    affine_path = datapath + '/' + 'diff2struct_affine.mat'

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    sigmas = [3.0, 1.0, 0.0]
    level_iters = [10000, 1000, 100]
    factors = [4, 2, 1]
    affreg_diff2struct = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = AffineTransform3D()
    params0 = None

    affine_diff2struct = affreg_diff2struct.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=None)

    saveAffineMat(affine_diff2struct, affine_path)


    # struct2standard affine registartion

    moving = t1_img
    moving_grid2world = t1_affine
    static = template_img
    static_grid2world = template_affine

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    sigmas = [3.0, 1.0, 0.0]
    level_iters = [10000, 1000, 100]
    factors = [4, 2, 1]
    affreg_struct2standard = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = AffineTransform3D()
    params0 = None
    affine_struct2standard = affreg_struct2standard.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=None)

    # struct2standard SyN registartion
    pre_align = affine_struct2standard.get_affine()
    metric = CCMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(static, moving, static_grid2world, moving_grid2world, pre_align)

    warped = mapping.transform_inverse(template_img)
    warped = affine_diff2struct.transform_inverse(warped)
    template_diff_path = datapath + '/' + 'MNI152_diff'
    save_nifti(template_diff_path, warped, affine_diff2struct.moving_grid2world)

    warped_roi = mapping.transform_inverse(roi_img)
    warped_roi = affine_diff2struct.transform_inverse(warped_roi)
    roi_diff_path = datapath + '/' + roi + '_diff.nii.gz'
    save_nifti(roi_diff_path, warped_roi, affine_diff2struct.moving_grid2world)

    print("  Done!  ")
