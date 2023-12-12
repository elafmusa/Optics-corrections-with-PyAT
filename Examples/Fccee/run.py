#!/usr/bin/env python
# coding: utf-8

# In[1]:


import at
import os
import scipy.io
from at_modules_loco import *
from data_loader import load_data, set_parameters,load_indices
from multiprocessing import Pool
output_root = './output'

error_folder = os.path.join(output_root, 'error')
correction_folder = os.path.join(output_root, 'correction')
os.makedirs(error_folder, exist_ok=True)
os.makedirs(correction_folder, exist_ok=True)

num_seeds = list(range(1, 51)) # 5 seeds
ring = at.load_mat('FCCee_z_566_nosol_4_bb')
ring.radiation_off()
num_used_correctors = 20

#Loading data & parameters
dCx_s, dCy_s, dCxy_s, dCyx_s, dCx, dCy, dCxy, dCyx, dCxs, dCys, dCxys, dCyxs, Rvx, Rvy = load_data()
num_used_correctors, correctors_kick, dk, numberOfIteration, sCut = set_parameters()
quads_indices, bpm_indices, corrector_indices, dipole_indices, used_cor_indices, sext_indexes, skew_quad, q_noskew, arc_quads, ir_quads, arc_sext,  ir_sext , nominal_crom, nominal_tune = load_indices(ring, num_used_correctors)

# Misalgined elemnts

misaligned_elements = arc_quads
misaligned_elements2 = arc_sext

shiftx = 10.e-6
shifty = 10.e-6
sigmaCut = 2.5 #sigma error cut
failed_seeds = 0
t0 = time.time()
for i in num_seeds:
    try:
        print(f"--------------------------------Seed_{i}---------------------------------------")
        ring = at.load_mat('FCCee_z_566_nosol_4_bb')
        ring.radiation_off()
        Cx0_on, Cxy0_on, Cy0_on, Cyx0_on = ORMs(correctors_kick, ring,used_cor_indices)

        print(' Turn sext off ')
        sext_strengths = []
        for n in sext_indexes:
            sext_strengths.append(ring[n].H)
            ring[n].H = 0.0

        [elemdata0, beamdata, elemdata] = at.get_optics(ring, bpm_indices)
        twiss = elemdata
        nominal_tune = get_tune(ring, get_integer=True)
        print(" BPM_to_corrector_response_matrices (Model) ..")
        Cx0, Cxy0, Cy0, Cyx0 = ORMs(correctors_kick, ring,used_cor_indices)

        print(" Introduce errors .. ")

        simulateShiftErrors(ring, shiftx,shifty, misaligned_elements, sigmaCut,  relative=True)
        simulateShiftErrors(ring, shiftx,shifty, misaligned_elements2, sigmaCut,  relative=True)
        simulateShiftErrors(ring, shiftx,shifty, dipole_indices, sigmaCut,  relative=True)

        filename = f'seed_{i}.mat'
        output_file_path = os.path.join(error_folder, filename)
        ring.save(output_file_path, mat_key='ring')


        print('SVD orbit correction')
        _, _, elemdata = at.get_optics(ring, bpm_indices)
        closed_orbitx = elemdata.closed_orbit[:, 0]
        closed_orbity = elemdata.closed_orbit[:, 2]
        CSx = np.dot(np.transpose(Rvx),-closed_orbitx)
        CSy = np.dot(np.transpose(Rvy),-closed_orbity)
        for m in range(len(corrector_indices)):
            ring[corrector_indices[m]].KickAngle  = [0,0]
            ring[corrector_indices[m]].KickAngle  = [CSx[m],CSy[m]]


        print("Swich sextupoles ON")
        for a in range(len(sext_indexes)):
            ring[sext_indexes[a]].H = sext_strengths[a]*1

        fit_tune(ring, get_refpts(ring, 'QF*'),
        get_refpts(ring, 'QD*'),nominal_tune )
        fit_chrom(ring, get_refpts(ring, 'SF*'),
        get_refpts(ring, 'SD*'),nominal_crom)

        print('LOCO iterations')
        print('Beta Beating Correction')
        for x in range(numberOfIteration):
            _, _, twiss_err = at.get_optics(ring, bpm_indices)
            Cx, Cxy, Cy, Cyx = ORMs(correctors_kick, ring, used_cor_indices)
            bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot=False)
            dx_rms_err, dy_rms_err = getDispersionErr(ring, twiss, bpm_indices, makeplot=False)
            A, B = defineJacobianMatrices_(Cx0, Cy0, Cx, Cy, dCx, dCy)
            Nk = len(dCx)
            r = getInverse(A, B, Nk, sCut, showPlots=False)
            setCorrection(ring, r, q_noskew)
            _, _, twiss_cor = at.get_optics(ring, bpm_indices)
            print('Beta beating before correction :')
            print("RMS horizontal beta beating:" + str(bx_rms_err * 100) + "%   RMS vertical beta beating: " + str(
                by_rms_err * 100) + "%")
            print('Beta beating after corrections :')
            bx_rms_cor, by_rms_cor = getBetaBeat(ring, twiss, bpm_indices, makeplot=False)
            print("RMS horizontal beta beating:" + str(bx_rms_cor * 100) + "%   RMS vertical beta beating: " + str(
                by_rms_cor * 100) + "%")
            # print('beta_x correction reduction', (1-bx_rms_cor/bx_rms_err)*100)
            # print('beta_x correction reduction', (1-by_rms_cor/by_rms_err)*100)
            # print('========================================')
            # print('dispersion')
            # print('========================================')
            print('dispersion before correction :')
            print("RMS horizontal dispersion:" + str(dx_rms_err) + "mm   RMS vertical dispersion: " + str(
                dy_rms_err) + "mm")
            print('dispersion after corrections')
            dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot=False)
            print("RMS horizontal dispersion:" + str(dx_rms_cor) + "mm   RMS vertical dispersion: " + str(
                dy_rms_cor) + "mm")
            # print('dispersion_x correction reduction', (1-dx_rms_cor/dx_rms_err)*100)
            # print('dispersion_y correction reduction', (1-dy_rms_cor/dy_rms_err)*100)

            # corrections_plots(ring ,twiss, twiss_err, 'beta')
            # corrections_plots(ring ,twiss, twiss_err, 'eta')

            _, _, twiss_err = at.get_optics(ring, bpm_indices)

            print('========================================')
            print('Dispersion Correction')
            print('========================================')
       #for x in range(numberOfIteration):
            # fit_tune(ring, get_refpts(ring, 'QF*'),
            # get_refpts(ring, 'QD*'),nominal_tune)
            # fit_chrom(ring, get_refpts(ring, 'SF*'),
            # get_refpts(ring, 'SD*'),nominal_crom)
            Cx, Cxy, Cy, Cyx = ORMs(correctors_kick, ring, used_cor_indices)
            bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot=False)
            dx_rms_err, dy_rms_err = getDispersionErr(ring, twiss, bpm_indices, makeplot=False)
            A, B = defineJacobianMatrices1(Cx0, Cy0, Cxy0, Cyx0, Cx, Cy, Cxy, Cyx, dCxs, dCys, dCxys, dCyxs)
            Nk = len(dCxs)
            r = getInverse(A, B, Nk, sCut, showPlots=False)
            setCorrection(ring, r[:len(q_noskew)], q_noskew)
            setCorrection(ring, r[len(q_noskew):], sext_indexes)
            _, _, twiss_cor = at.get_optics(ring, bpm_indices)
            print('Beta beating before correction :')
            print("RMS horizontal beta beating:" + str(bx_rms_err * 100) + "%   RMS vertical beta beating: " + str(
                by_rms_err * 100) + "%")
            print('Beta beating after corrections :')
            bx_rms_cor, by_rms_cor = getBetaBeat(ring, twiss, bpm_indices, makeplot=False)
            print("RMS horizontal beta beating:" + str(bx_rms_cor * 100) + "%   RMS vertical beta beating: " + str(
                by_rms_cor * 100) + "%")
            print('beta_x correction reduction', (1 - bx_rms_cor / bx_rms_err) * 100)
            print('beta_x correction reduction', (1 - by_rms_cor / by_rms_err) * 100)
            # print('========================================')
            # print('dispersion')
            # print('========================================')
            print('dispersion before correction :')
            print("RMS horizontal dispersion:" + str(dx_rms_err) + "mm   RMS vertical dispersion: " + str(
                dy_rms_err) + "mm")
            print('dispersion after corrections')
            dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot=False)
            print("RMS horizontal dispersion:" + str(dx_rms_cor) + "mm   RMS vertical dispersion: " + str(
                dy_rms_cor) + "mm")
            print('dispersion_x correction reduction', (1 - dx_rms_cor / dx_rms_err) * 100)
            print('dispersion_y correction reduction', (1 - dy_rms_cor / dy_rms_err) * 100)

        filename = f'seed_{i}.mat'
        output_file_path = os.path.join(correction_folder, filename)
        ring.save(output_file_path, mat_key='ring')
    except Exception as e:
        print(f"An error occurred for Seed_{i}: {str(e)}")
        failed_seeds += 1
        continue


        filename = f'seed_{i}.mat'
        output_file_path = os.path.join(correction_folder, filename)
        ring.save(output_file_path, mat_key='ring')
    except Exception as e:
        print(f"An error occurred for Seed_{i}: {str(e)}")
        failed_seeds += 1
        continue

t1 = time.time()
print(f"Execution time For: {len(num_seeds)} seeds = {(t1 - t0) / 3600} hours")
print(f"Number of failed seeds: {failed_seeds}")


# In[ ]:





# In[ ]:





# In[ ]:




