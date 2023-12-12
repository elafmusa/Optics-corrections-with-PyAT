import numpy as np
import at
from at_modules_loco import *


def load_data():
    dCx = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCx_of_106_2.npy')
    dCy = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCy_of_106_2.npy')
    dCxy = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCxy_of_106_2.npy')
    dCyx = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCyx_of_106_2.npy')
    dCx_s = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCx_of_106_skew_2.npy')
    dCy_s = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCy_of_106_skew_2.npy')
    dCxy_s = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCxy_of_106_skew_2.npy')
    dCyx_s = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\dCyx_of_106_skew_2.npy')

    dCxs = np.concatenate((dCx, dCx_s), axis=0)
    dCys = np.concatenate((dCy, dCy_s), axis=0)
    dCxys = np.concatenate((dCxy, dCxy_s), axis=0)
    dCyxs = np.concatenate((dCyx, dCyx_s), axis=0)



    Rvx = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\Rvx_566_2.npy')
    Rvy = np.load(r'C:\Users\musa\Desktop\at\My_Examples\566-cor\matrices_jacobians\Rvy_566_2.npy')

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
    cor_x = get_refpts(ring, 'CX*')
    cor_y = get_refpts(ring, 'CY*')
    bpm_x = [i+1 for i in cor_x]
    bpm_y = [i+1 for i in cor_y]

    # ORM
    print("Calculate ORM before errors (Model)")
    used_cor_x = select_equally_spaced_elements(cor_x,num_used_correctors)
    used_cor_y = select_equally_spaced_elements(cor_y, num_used_correctors)

    #used_cor_indices = select_equally_spaced_elements(corrector_indices, num_used_correctors)
    nominal_crom = get_chrom(ring)
    nominal_tune = get_tune(ring, get_integer=True)
    sext_indexes = get_refpts(ring, at.elements.Sextupole)
    skew_quad = get_refpts(ring, "skew*")
    arc_dipoles = get_refpts(ring, "B1*")
    ir_dipoles = [i for i in dipole_indices if i not in arc_dipoles]

    q_noskew = [i for i in quads_indices if i not in skew_quad]
    arc_quads = [i for i in quads_indices if re.match(r'q[df][1-4]', ring[i].FamName)]
    ir_sext = [
        i
        for i in sext_indexes
        if (re.match(r's[fd][1-2][.]', ring[i].FamName)
            or re.match(r'sy', ring[i].FamName))]
    arc_sext = [i for i in sext_indexes if i not in ir_sext]
    ir_quads = [i for i in quads_indices if i not in arc_quads]

    qf = []
    qd = []

    for i in arc_quads:
        if ring[i].FamName.startswith('qf'):
            qf.append(i)
        elif ring[i].FamName.startswith('qd'):
            qd.append(i)

    sf = []
    sd = []

    for i in arc_sext:
        if ring[i].FamName.startswith('sf'):
            sf.append(i)
        elif ring[i].FamName.startswith('sd'):
            sd.append(i)

    return quads_indices, bpm_indices, corrector_indices, dipole_indices, used_cor_x, used_cor_y, \
         sext_indexes, skew_quad, q_noskew, arc_quads, ir_quads, arc_sext,  ir_sext ,arc_dipoles,ir_dipoles,  nominal_crom, nominal_tune, qf,qd,sf,sd
