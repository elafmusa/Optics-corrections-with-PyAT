#!/usr/bin/env python
# coding: utf-8

# In[1]:


import at
import os
from at_modules_loco import *
from data_loader import load_data, set_parameters,load_indices
output_root = '.\output'

error_folder = os.path.join(output_root, 'error')
error_folder2 = os.path.join(output_root, 'error2')
correction_folder = os.path.join(output_root, 'correction')
correction_folder2 = os.path.join(output_root, 'correction2')


os.makedirs(error_folder, exist_ok=True)
os.makedirs(correction_folder, exist_ok=True)
os.makedirs(error_folder2, exist_ok=True)
os.makedirs(correction_folder2, exist_ok=True)

num_seeds = list(range(1, 3)) # 51 seeds
ring = at.load_mat('FCCee_z_566_corx_cory_bpmq_skew')
ring.radiation_off()
num_used_correctors = 20

#Loading data & parameters

dCx_s, dCy_s, dCxy_s, dCyx_s, dCx, dCy, dCxy, dCyx, dCxs, dCys, dCxys, dCyxs, Rvx, Rvy = load_data()
num_used_correctors, correctors_kick, dk, numberOfIteration, sCut = set_parameters()
quads_indices, bpm_indices, corrector_indices, dipole_indices, used_cor_x, used_cor_y, sext_indexes, skew_quad, q_noskew, arc_quads, ir_quads, arc_sext,  ir_sext , arc_dipoles, ir_dipoles, nominal_crom, nominal_tune, qf,qd,sf,sd = load_indices(ring, num_used_correctors)

# Misalgined elemnts

cor_x = get_refpts(ring, 'CX*')
cor_y = get_refpts(ring, 'CY*')
misaligned_elements = arc_quads
misaligned_elements2 = arc_sext

sigma_err = 50.e-6
shiftx = sigma_err
shifty = sigma_err
tilts = sigma_err
pitches = sigma_err
yaws = sigma_err
#gradErr = 1.e-4
sigmaCut = 2.5 #sigma error cut
failed_seeds = 0
t0 = time.time()
for i in num_seeds:
    try:
        print(f"--------------------------------Seed_{i}---------------------------------------")
        ring = at.load_mat('FCCee_z_566_corx_cory_bpmq_skew')
        ring.radiation_off()
        #Cx0_on, Cxy0_on, Cy0_on, Cyx0_on = ORMs(correctors_kick, ring,used_cor_indices)

        Cx0_on, Cxy0_on = ORM_x1(correctors_kick, ring, used_cor_x, bpm_indices)
        Cy0_on, Cyx0_on = ORM_y1(correctors_kick, ring, used_cor_y, bpm_indices)

        print('---Turn sext off---')
        sext_strengths = []
        for n in sext_indexes:
            sext_strengths.append(ring[n].H)
            ring[n].H = 0.0

        [elemdata0, beamdata, elemdata] = at.get_optics(ring, bpm_indices)
        twiss = elemdata
        print(" BPM_to_corrector_response_matrices (Model) ..")

        Cx0 ,Cxy0 = ORM_x1(correctors_kick, ring, used_cor_x, bpm_indices)
        Cy0, Cyx0 = ORM_y1(correctors_kick, ring, used_cor_y, bpm_indices)

        print(" Introduce errors .. ")


        simulateShiftErrors(ring, shiftx,shifty, misaligned_elements, sigmaCut,  relative=True)
        simulateShiftErrors(ring, shiftx,shifty, misaligned_elements2, sigmaCut,  relative=True)
        simulateShiftErrors(ring, shiftx,shifty, dipole_indices, sigmaCut,  relative=True)

        #simulateTilttErrors(ring, tilts,pitches, yaws,misaligned_elements , sigmaCut, relative=True)
        #simulateTilttErrors(ring, tilts,pitches, yaws,misaligned_elements2 , sigmaCut, relative=True)
        #simulateTilttErrors(ring, tilts,0, 0,dipole_indices , sigmaCut, relative=True)

        rmsx, rmsy = rms_orbits(ring, bpm_indices, makeplot = False)
        bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
        print(f"RMS horizontal orbit with errors: {rmsx*1.e6} mu m, ", f"RMS vertical orbit with errors: {rmsy*1.e6} mu m")
        print("RMS horizontal beta beating:" + str(bx_rms_err* 100) + "%   RMS vertical beta beating: " + str(by_rms_err* 100) + "%")
        dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
        print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
        print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
              f"The chromaticity values: {get_chrom(ring)}. ")

        filename = f'seed_{i}.mat'
        output_file_path = os.path.join(error_folder, filename)
        ring.save(output_file_path, mat_key='ring')


        print('---SVD orbit correction---')
        rmsx1, rmsy1 = rms_orbits(ring, bpm_indices, makeplot = False)

        _, _, elemdata = at.get_optics(ring, bpm_indices)
        closed_orbitx = elemdata.closed_orbit[:, 0]
        _, _, elemdata = at.get_optics(ring, bpm_indices)
        closed_orbity = elemdata.closed_orbit[:, 2]
        CSx = np.dot(np.transpose(Rvx), -closed_orbitx)
        CSy = np.dot(np.transpose(Rvy), -closed_orbity)

        for m in range(len(cor_x)):
            ring[cor_x[m]].KickAngle  = [0,0]
            ring[cor_x[m]].KickAngle  = [CSx[m],0]

        for m in range(len(cor_y)):
            ring[cor_y[m]].KickAngle  = [0,0]
            ring[cor_y[m]].KickAngle  = [0,CSy[m]]

        rmsx, rmsy = rms_orbits(ring, bpm_indices, makeplot = False)
        bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
        print(f"RMS horizontal orbit with errors: {rmsx1*1.e6} mu m, ", f"RMS vertical orbit with errors: {rmsy1*1.e6} mu m")
        print(f"RMS horizontal orbit after correction: {rmsx*1.e6} mu m, ", f"RMS vertical orbit after correction: {rmsy*1.e6} mu m")
        print("RMS horizontal beta beating:" + str(bx_rms_err* 100) + "%   RMS vertical beta beating: " + str(by_rms_err* 100) + "%")
        dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
        print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
        print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
              f"The chromaticity values: {get_chrom(ring)}. ")

        print("----Swich sextupoles ON----")

        for a in range(len(sext_indexes)):
            ring[sext_indexes[a]].H = sext_strengths[a]*1

        filename = f'seed_{i}_sextON.mat'
        output_file_path = os.path.join(error_folder2, filename)
        ring.save(output_file_path, mat_key='ring')

        rmsx, rmsy = rms_orbits(ring, bpm_indices, makeplot = False)
        bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
        #print(f"RMS horizontal orbit with errors: {rmsx1*1.e6} mu m, ", f"RMS vertical orbit with errors: {rmsy1*1.e6} mu m")
        print(f"RMS horizontal orbit after correction: {rmsx*1.e6} mu m, ", f"RMS vertical orbit after correction: {rmsy*1.e6} mu m")
        print("RMS horizontal beta beating:" + str(bx_rms_err* 100) + "%   RMS vertical beta beating: " + str(by_rms_err* 100) + "%")
        dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
        print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
        print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
              f"The chromaticity values: {get_chrom(ring)}. ")

        print("---Fit Tune & Chrom---")
        fit_tune(ring, qf,
        qd,nominal_tune)
        fit_chrom(ring, sf,
        sd,nominal_crom)


        rmsx, rmsy = rms_orbits(ring, bpm_indices, makeplot = False)
        bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
        #print(f"RMS horizontal orbit with errors: {rmsx1*1.e6} mu m, ", f"RMS vertical orbit with errors: {rmsy1*1.e6} mu m")
        print(f"RMS horizontal orbit after correction: {rmsx*1.e6} mu m, ", f"RMS vertical orbit after correction: {rmsy*1.e6} mu m")
        print("RMS horizontal beta beating:" + str(bx_rms_err* 100) + "%   RMS vertical beta beating: " + str(by_rms_err* 100) + "%")
        dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
        print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
        print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
              f"The chromaticity values: {get_chrom(ring)}. ")

        print('---LOCO iterations---')
        _, _, twiss_err0 = at.get_optics(ring, bpm_indices)

        numberOfIteration = 2
        for x in range(numberOfIteration):
            print('---Beta Beating Correction---')
            _, _, twiss_err = at.get_optics(ring, bpm_indices)
            Cx ,Cxy = ORM_x1(correctors_kick, ring, used_cor_x, bpm_indices)
            Cy, Cyx = ORM_y1(correctors_kick, ring, used_cor_y, bpm_indices)

            bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
            dx_rms_err, dy_rms_err = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
            A, B = defineJacobianMatrices_(Cx0, Cy0, Cx, Cy, dCx, dCy)
            Nk = len(dCx)
            r = getInverse(A, B, Nk, 1250, showPlots = False)
            setCorrection(ring, r , q_noskew)
            _, _, twiss_cor = at.get_optics(ring, bpm_indices)
            print('Beta beating before correction :')
            print("RMS horizontal beta beating:" + str(bx_rms_err * 100) + "%   RMS vertical beta beating: " + str(by_rms_err * 100) + "%")
            print('Beta beating after corrections :')
            bx_rms_cor, by_rms_cor = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
            print("RMS horizontal beta beating:" + str(bx_rms_cor * 100) + "%   RMS vertical beta beating: " + str(by_rms_cor * 100) + "%")
            print('beta_x correction reduction', (1-bx_rms_cor/bx_rms_err)*100)
            print('beta_x correction reduction', (1-by_rms_cor/by_rms_err)*100)

            rmsx, rmsy = rms_orbits(ring, bpm_indices, makeplot = False)
            print(f"RMS horizontal orbit after correction: {rmsx*1.e6} mu m, ", f"RMS vertical orbit after correction: {rmsy*1.e6} mu m")
            print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
                  f"The chromaticity values: {get_chrom(ring)}. ")
            #print('========================================')
            #print('dispersion')
            #print('========================================')
            print('dispersion before correction :')
            print("RMS horizontal dispersion:" + str(dx_rms_err ) + "mm   RMS vertical dispersion: " + str(dy_rms_err) + "mm")
            print('dispersion after corrections')
            dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
            print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
            print('dispersion_x correction reduction', (1-dx_rms_cor/dx_rms_err)*100)
            print('dispersion_y correction reduction', (1-dy_rms_cor/dy_rms_err)*100)

            #corrections_plots(ring ,twiss, twiss_err0, 'beta')
            #corrections_plots(ring ,twiss, twiss_err0, 'eta')
            _, _, twiss_err = at.get_optics(ring, bpm_indices)

        filename = f'seed_{i}_betaCor.mat'
        output_file_path = os.path.join(correction_folder2, filename)
        ring.save(output_file_path, mat_key='ring')

        numberOfIteration = 1
        for x in range(numberOfIteration):

            print("---Fit Tune & Chrom---")
            fit_tune(ring, qf,
            qd,nominal_tune)
            fit_chrom(ring, sf,
            sd,nominal_crom)
            print('---- Coupling & Dispersion Correction ----')

            Cx ,Cxy = ORM_x1(correctors_kick, ring, used_cor_x, bpm_indices)
            Cy, Cyx = ORM_y1(correctors_kick, ring, used_cor_y, bpm_indices)
            bx_rms_err, by_rms_err = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
            dx_rms_err, dy_rms_err = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
            A, B = defineJacobianMatrices1(Cx0, Cy0, Cxy0, Cyx0, Cx, Cy, Cxy, Cyx, dCxs, dCys, dCxys,dCyxs)
            Nk = len(dCxs)
            r = getInverse(A, B, Nk, 1500, showPlots = False)
            setCorrection(ring, r[:len(q_noskew)] , q_noskew)
            setCorrection(ring, r[len(q_noskew):] , skew_quad)
            _, _, twiss_cor = at.get_optics(ring, bpm_indices)
            print('Beta beating before correction :')
            print("RMS horizontal beta beating:" + str(bx_rms_err * 100) + "%   RMS vertical beta beating: " + str(by_rms_err * 100) + "%")
            print('Beta beating after corrections :')
            bx_rms_cor, by_rms_cor = getBetaBeat(ring, twiss, bpm_indices, makeplot = False)
            print("RMS horizontal beta beating:" + str(bx_rms_cor * 100) + "%   RMS vertical beta beating: " + str(by_rms_cor * 100) + "%")
            print('beta_x correction reduction', (1-bx_rms_cor/bx_rms_err)*100)
            print('beta_x correction reduction', (1-by_rms_cor/by_rms_err)*100)
            #print('========================================')
            #print('dispersion')
            #print('========================================')
            print('Dispersion before correction :')
            print("RMS horizontal dispersion:" + str(dx_rms_err ) + "mm   RMS vertical dispersion: " + str(dy_rms_err) + "mm")
            print('Dispersion after corrections')
            dx_rms_cor, dy_rms_cor = getDispersionErr(ring, twiss, bpm_indices, makeplot = False)
            print("RMS horizontal dispersion:" + str(dx_rms_cor ) + "mm   RMS vertical dispersion: " + str(dy_rms_cor) + "mm")
            print('dispersion_x correction reduction', (1-dx_rms_cor/dx_rms_err)*100)
            print('dispersion_y correction reduction', (1-dy_rms_cor/dy_rms_err)*100)

            print(f"RMS horizontal orbit after correction: {rmsx*1.e6} mu m, ", f"RMS vertical orbit after correction: {rmsy*1.e6} mu m")
            print(f"Tune values with orbit corrections: {get_tune(ring, get_integer=True)}, "
                  f"The chromaticity values: {get_chrom(ring)}. ")

            #corrections_plots(ring ,twiss, twiss_err0, 'beta')
            #corrections_plots(ring ,twiss, twiss_err0, 'eta')

            #print("---Fit Tune & Chrom---")
            #fit_tune(ring, qf,
            #qd,nominal_tune , KStep = 1e-9)
            #fit_chrom(ring, sf,
            #sd,nominal_crom,  HStep=1e-8)


        filename = f'seed_{i}_all.mat'
        output_file_path = os.path.join(correction_folder, filename)
        ring.save(output_file_path, mat_key='ring')
    except Exception as e:
        print(f"An error occurred for Seed_{i}: {str(e)}")
        failed_seeds += 1
        continue

        #ring.radiation_on()
        ring.enable_6d()
        ring.tapering(niter=3, quadrupole=True, sextupole=True)
        emit0, bbb, eee = ring.ohmi_envelope()
        emittance_h = emit0['emitXY'][0]
        emittance_v = emit0['emitXY'][1]
        print('emittance_h', emittance_h *1e9, 'emittance_v',emittance_v*1e12)
        emittance_h/emittance_v


t1 = time.time()
print(f"Execution time For: {len(num_seeds)} seeds = {(t1 - t0) / 3600} hours")
print(f"Number of failed seeds: {failed_seeds}")


# In[ ]:




