import numpy as np

from at_modules_loco import *


def load_data(ring):
    C_measured = np.transpose(np.loadtxt("ORM_240208.txt"))
    Noise_BPMx = np.loadtxt("Noise_240802_BPMx.txt")
    Noise_BPMy = np.loadtxt("Noise_240802_BPMy.txt")

    BPM_names = load_names("BPM_names.txt")
    HCM_names = load_names("HCM_names.txt")
    VCM_names = load_names("VCM_names.txt")

    Jacobian = np.load('p3_Jacobian.npy')
    C_model = np.load('C_model.npy')

    return C_measured, C_model, Noise_BPMx, Noise_BPMy, BPM_names, HCM_names, VCM_names, Jacobian,


def load_indices(ring):
    quads_indices = at.get_refpts(ring, at.elements.Quadrupole)
    bpm_indices = at.get_refpts(ring, at.elements.Monitor)
    bpm_x_ind = bpm_indices
    bpm_y_ind = bpm_indices
    corrector_indices = at.get_refpts(ring, at.elements.Corrector)
    dipole_indices = at.get_refpts(ring, at.elements.Dipole)
    skew_indices = at.get_refpts(ring, "QS*")
    sext_indexes = at.get_refpts(ring, at.elements.Sextupole)
    HCM_names = load_names("HCM_names.txt")
    VCM_names = load_names("VCM_names.txt")

    hcm_index = []
    for i in HCM_names:
        hcm_index.append(at.get_refpts(ring, i))
    vcm_index = []
    for i in VCM_names:
        vcm_index.append(at.get_refpts(ring, i))

    hcm_index = [index[0] for index in hcm_index]
    vcm_index = [index[0] for index in vcm_index]

    nominal_crom = get_chrom(ring)
    nominal_tune = get_tune(ring, get_integer=True)

    data = load_names("orm_correction_5141_1.txt")
    quad_fam_names = extract_names_with_q(str(data))

    quad_fam_indices = []
    for i in quad_fam_names:
        quad_fam_indices.append(at.get_refpts(ring, i))



    return quads_indices, skew_indices, quad_fam_names, quad_fam_indices, bpm_indices, corrector_indices, dipole_indices,sorted(hcm_index),sorted(vcm_index) ,sext_indexes, bpm_x_ind, bpm_y_ind, nominal_crom, nominal_tune
