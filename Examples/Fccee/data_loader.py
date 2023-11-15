import numpy as np
import at
from at_modules_loco import *


def load_data():
    dCx_s = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCx_on_sext106.npy')
    dCy_s = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCy_on_sext106.npy')
    dCxy_s = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCxy_on_sext106.npy')
    dCyx_s = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCyx_on_sext106.npy')
    dCx = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCx_on_normal106.npy')
    dCy = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCy_on_normal106.npy')
    dCxy = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCxy_on_normal106.npy')
    dCyx = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\dCyx_on_normal106.npy')
    dCxs = np.concatenate((dCx, dCx_s), axis=0)
    dCys = np.concatenate((dCy, dCy_s), axis=0)
    dCxys = np.concatenate((dCxy, dCxy_s), axis=0)
    dCyxs = np.concatenate((dCyx, dCyx_s), axis=0)
    Rvx = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\Rvx_z_new.npy')
    Rvy = np.load(r'C:\Users\musa\Desktop\FCC-PyAT\Seeds\Rvy_z_new.npy')

    return dCx_s, dCy_s, dCxy_s, dCyx_s, dCx, dCy, dCxy, dCyx, dCxs, dCys, dCxys, dCyxs, Rvx, Rvy


def set_parameters():
    num_used_correctors = 20
    correctors_kick = 1.e-10
    dk = 1.e-6
    numberOfIteration = 2
    sCut = 1250
    # Set other parameters here and return them as needed
    return num_used_correctors, correctors_kick, dk, numberOfIteration, sCut


import at


def load_indices(ring, num_used_correctors):
    quads_indices = get_refpts(ring, at.elements.Quadrupole)
    bpm_indices = get_refpts(ring, at.elements.Monitor)
    corrector_indices = get_refpts(ring, at.elements.Corrector)
    dipole_indices = get_refpts(ring, elements.Dipole)
    used_cor_indices = select_equally_spaced_elements(corrector_indices, num_used_correctors)
    nominal_crom = get_chrom(ring)
    nominal_tune = get_tune(ring, get_integer=True)
    sext_indexes = get_refpts(ring, at.elements.Sextupole)
    skew_quad = get_refpts(ring, "skew*")
    q_noskew = [i for i in quads_indices if i not in skew_quad]
    arc_quads = [i for i in quads_indices if re.match(r'q[df][1-4]', ring[i].FamName)]
    ir_sext = [
        i
        for i in sext_indexes
        if (re.match(r's[fd][1-2][.]', ring[i].FamName)
            or re.match(r'sy', ring[i].FamName))]
    arc_sext = [i for i in sext_indexes if i not in ir_sext]
    ir_quads = [i for i in quads_indices if i not in arc_quads]

    return quads_indices, bpm_indices, corrector_indices, dipole_indices, used_cor_indices, \
         sext_indexes, skew_quad, q_noskew, arc_quads, ir_quads, arc_sext,  ir_sext , nominal_crom, nominal_tune
