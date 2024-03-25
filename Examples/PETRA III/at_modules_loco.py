import at
from pylab import *
from at import *
import at.plot
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd



def calculate_jacobian(ring, quads_ind, C_model, dkick, used_cor_ind_h, used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, dk, includeDispersion=False, include_coupling=True, skew=False, family=True, full_jacobian=True):
    """
    Calculates the Jacobian matrix for the given parameters and options.
    """

    args_list = [
        (quad_index, ring, C_model, dkick, used_cor_ind_h, used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, dk, includeDispersion, include_coupling, skew, family)
        for quad_index in quads_ind
    ]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(generating_quads_response_matrices, args_list)

    results = [result / dk for result in results]

    if full_jacobian:
        n_correctors = len(used_cor_ind_h) + len(used_cor_ind_v)
        n_bpms = len(bpm_indexes_h) + len(bpm_indexes_v)
        j_cor = np.zeros((n_correctors,) + C_model.shape)
        for i in range(n_correctors):
            j_cor[i, i, :] = C_model[i, :]  # a single column of response matrix corresponding to a corrector
        j_bpm = np.zeros((n_bpms,) + C_model.shape)
        for i in range(n_bpms):
            j_bpm[i, :, i] = C_model[:, i]  # a single row of response matrix corresponding to a given plane of BPM
        return np.concatenate((results, j_cor, j_bpm), axis=0)

    return results

def generating_quads_response_matrices(quad_index, ring, C_model, dkick, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, dk
                 , includeDispersion, include_coupling, skew, family):
    #i = quad_index

    if family==False:

        print('generating response to quad of index', quad_index)

        l = ring[quad_index].Length

        if skew == True:
            strength_before = ring[quad_index].PolynomA[1]
        else:
            strength_before = ring[quad_index].PolynomB[1]

        if skew == True:
            ring[quad_index].PolynomA[1] = strength_before + dk
        else:
            ring[quad_index].PolynomB[1] = strength_before + dk

        C_simulate = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)

        if skew == True:
            ring[quad_index].PolynomA[1] = strength_before
        else:
            ring[quad_index].PolynomB[1] = strength_before

        return (C_simulate - C_model)   / l


    else:

        #q_families = quad_index
        #for quads in range(len(q_families)):

        q_family = quad_index
        l = ring[q_family[0]].Length
        name = ring[q_family[0]].FamName

        strength_before = []

        print('generating response to Fam {}, n={}'.format(name, len(q_family)))
        for quad in range(len(q_family)):

            if skew == True:
                strength_before.append(ring[q_family[quad]].PolynomA[1])
            else:
                strength_before.append(ring[q_family[quad]].PolynomB[1])

            if skew == True:
                ring[q_family[quad]].PolynomA[1] = strength_before[quad]  + dk
            else:
                ring[q_family[quad]].PolynomB[1] = strength_before[quad] +  dk

        C_simulate = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)

        for quad in range(len(q_family)):

            if skew == True:
                ring[q_family[quad]].PolynomA[1] = strength_before[quad]
            else:
                ring[q_family[quad]].PolynomB[1] = strength_before[quad]


        return (C_simulate - C_model) / l

def generating_quads_response_matrices_2(quad_index, ring, C_model, dkick, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, dk
                 , includeDispersion, include_coupling, skew, family):
    #i = quad_index

    if family==False:

        print('generating response to quad of index', quad_index)

        l = ring[quad_index].Length

        if skew == True:
            strength_before = ring[quad_index].PolynomA[1]
        else:
            strength_before = ring[quad_index].PolynomB[1]

        if skew == True:
            ring[quad_index].PolynomA[1] = strength_before + dk / 2
        else:
            ring[quad_index].PolynomB[1] = strength_before + dk / 2

        C_simulate_p = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)


        if skew == True:
            ring[quad_index].PolynomA[1] = strength_before - dk / 2
        else:
            ring[quad_index].PolynomB[1] = strength_before - dk / 2

        C_simulate_n = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)

        C_simulate = C_simulate_p - C_simulate_n


        if skew == True:
            ring[quad_index].PolynomA[1] = strength_before
        else:
            ring[quad_index].PolynomB[1] = strength_before

        return (C_simulate - C_model) / l

    else:

        #q_families = quad_index
        #for quads in range(len(q_families)):

        q_family = quad_index
        l = ring[q_family[0]].Length
        name = ring[q_family[0]].FamName

        strength_before = []

        print('generating response to Fam {}, n={}'.format(name, len(q_family)))
        for quad in range(len(q_family)):

            if skew == True:
                strength_before.append(ring[q_family[quad]].PolynomA[1])
            else:
                strength_before.append(ring[q_family[quad]].PolynomB[1])

            if skew == True:
                ring[q_family[quad]].PolynomA[1] +=   dk / 2
            else:
                ring[q_family[quad]].PolynomB[1] +=   dk / 2

        C_simulatep = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)

        for quad in range(len(q_family)):

            if skew == True:
                strength_before.append(ring[q_family[quad]].PolynomA[1])
            else:
                strength_before.append(ring[q_family[quad]].PolynomB[1])

            if skew == True:
                ring[q_family[quad]].PolynomA[1] -=   dk / 2
            else:
                ring[q_family[quad]].PolynomB[1] -=   dk / 2

        C_simulaten = model_orm(dkick, ring, used_cor_ind_h ,used_cor_ind_v, bpm_indexes_h, bpm_indexes_v, includeDispersion, include_coupling)

        C_simulate = C_simulatep - C_simulaten

        for quad in range(len(q_family)):

            if skew == True:
                ring[q_family[quad]].PolynomA[1] = strength_before[quad]
            else:
                ring[q_family[quad]].PolynomB[1] = strength_before[quad]


        return (C_simulate - C_model) / l


def model_orm(dkick, ring, used_correctors_ind_h, used_correctors_ind_v, used_bpm_h,used_bpm_v, includeDispersion=False, include_coupling=True):
    Cx0, Cxy0 = ORM_x1(dkick, ring, used_correctors_ind_h, used_bpm_h, includeDispersion)
    Cy0, Cyx0 = ORM_y1(dkick, ring, used_correctors_ind_v, used_bpm_v, includeDispersion)

    if include_coupling:
        X = np.zeros((Cx0.shape[0] + Cy0.shape[0], Cx0.shape[1] + Cy0.shape[1]))
        X[:Cx0.shape[0], :Cx0.shape[1]] = Cx0
        X[:Cx0.shape[0], Cx0.shape[1]:] = Cxy0
        X[Cx0.shape[0]:, :Cx0.shape[1]] = Cyx0
        X[Cx0.shape[0]:, Cx0.shape[1]:] = Cy0
    else:
        # When coupling is not included, return only the diagonal blocks (Cx0, Cy0), ignoring coupling terms
        # Note: The output shape changes, it becomes a block diagonal matrix with zeroed coupling terms
        X = np.zeros((Cx0.shape[0] + Cy0.shape[0], Cx0.shape[1] + Cy0.shape[1]))
        X[:Cx0.shape[0], :Cx0.shape[1]] = Cx0
        X[Cx0.shape[0]:, Cx0.shape[1]:] = Cy0

    return X



def ORM_x(dkick, ring, used_correctors_ind, used_bpm,includeDispersion=False):
    cxx_p = []
    cxy_p = []
    cxx_m = []
    cxy_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        #cor_index = cor_index[0]

        a = ring[cor_index].KickAngle[0]
        ring[cor_index].KickAngle[0] = dkick/2 + a
        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        print('closed_orbitx1', closed_orbitx[:4])
        print('closed_orbity1', closed_orbity[:4])
        cxx_p.append(closed_orbitx)
        cxy_p.append(closed_orbity)

        ring[cor_index].KickAngle[0] = -dkick / 2 + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        print('closed_orbitx2', closed_orbitx[:4])
        print('closed_orbity2', closed_orbity[:4])
        cxx_m.append(closed_orbitx)
        cxy_m.append(closed_orbity)


        ring[cor_index].KickAngle[0] = a

    Cxx = (np.squeeze(cxx_p) - np.squeeze(cxx_m))
    Cxy = (np.squeeze(cxy_p) - np.squeeze(cxy_m))

    if includeDispersion==True:

        lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm)
        Eta_xx = lindata['dispersion'][:, 0]
        Eta_yy = lindata['dispersion'][:, 2]

        Cxx = np.vstack((Cxx, Eta_xx))
        Cxy = np.vstack((Cxy, Eta_yy))

    Cxx = Cxx / dkick
    Cxy = Cxy / dkick

    return (Cxx), (Cxy)

def ORM_y(dkick, ring, used_correctors_ind, used_bpm, includeDispersion=False):
    cyy_p = []
    cyx_p = []
    cyy_m = []
    cyx_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        #cor_index = cor_index[0]
        a = ring[cor_index].KickAngle[1]
        ring[cor_index].KickAngle[1] = dkick/2 + a
        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cyy_p.append(closed_orbity)
        cyx_p.append(closed_orbitx)

        ring[cor_index].KickAngle[1] = -dkick / 2 + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cyy_m.append(closed_orbity)
        cyx_m.append(closed_orbitx)


        ring[cor_index].KickAngle[1] = a

    Cyy = (np.squeeze(cyy_p) - np.squeeze(cyy_m))
    Cyx = (np.squeeze(cyx_p) - np.squeeze(cyx_m))

    if includeDispersion == True:
        lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm)
        Eta_xx = lindata['dispersion'][:, 0]
        Eta_yy = lindata['dispersion'][:, 2]

        Cyy = np.vstack((Cyy, Eta_yy))
        Cyx = np.vstack((Cyx, Eta_xx))

    Cyy = Cyy / dkick
    Cyx = Cyx / dkick

    return (Cyy), (Cyx)

def ORM_x1(dkick, ring, used_correctors_ind, used_bpm, includeDispersion =False):
    cxx_p = []
    cxy_p = []
    cxx_m = []
    cxy_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        #cor_index = cor_index[0]

        L0 = ring[cor_index].Length
        if L0 == 0:
            L = 1
        else:
            L = L0

        a = ring[cor_index].KickAngle[0]
        ring[cor_index].KickAngle[0] = (dkick/L) + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        print('closed_orbitx1', closed_orbitx[:4])
        print('closed_orbity1', closed_orbity[:4])
        cxx_p.append(closed_orbitx)
        cxy_p.append(closed_orbity)

        ring[cor_index].KickAngle[0] = -(dkick / L) + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        print('closed_orbitx2', closed_orbitx[:4])
        print('closed_orbity2', closed_orbity[:4])
        cxx_m.append(closed_orbitx)
        cxy_m.append(closed_orbity)


        ring[cor_index].KickAngle[0] = a
    Cxx = (np.squeeze(cxx_p)-np.squeeze(cxx_m))/2/dkick
    Cxy = (np.squeeze(cxy_p)-np.squeeze(cxy_m))/2/dkick

    return (Cxx), (Cxy)


def ORM_y1(dkick, ring, used_correctors_ind, used_bpm,includeDispersion =False):
    cyy_p = []
    cyx_p = []
    cyy_m = []
    cyx_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        #cor_index = cor_index[0]

        L0 = ring[cor_index].Length
        if L0 == 0:
           L = 1
        else:
           L = L0

        a = ring[cor_index].KickAngle[1]
        ring[cor_index].KickAngle[1] = (dkick/L) + a
        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cyy_p.append(closed_orbity)
        cyx_p.append(closed_orbitx)

        ring[cor_index].KickAngle[1] = -(dkick / L) + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cyy_m.append(closed_orbity)
        cyx_m.append(closed_orbitx)


        ring[cor_index].KickAngle[1] = a
    Cyy = (np.squeeze(cyy_p)-np.squeeze(cyy_m))/2/dkick
    Cyx = (np.squeeze(cyx_p)-np.squeeze(cyx_m))/2/dkick

    return (Cyy), (Cyx)


def loco_correction_lm(initial_guess0, orm_model, orm_measured, Jn, lengths, including_fit_parameters, bounds=(-np.inf, np.inf), weights=1,
                       verbose=2):
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    result = least_squares(lambda delta_params: objective(delta_params, orm_measured - orm_model, Jn[mask, :, :], weights),
                           initial_guess0[mask], #bounds=bounds,
                           method="lm",
                           verbose=verbose)
    return result.x




def loco_correction_ng_(initial_guess0, orm_model, orm_measured, J,Jt, lengths, including_fit_parameters, s_cut, weights=1, eps=1e-6,max_iterations =100):
    Iter = 0
    while True:
        Iter += 1

        if max_iterations is not None and Iter > max_iterations:
            break

        initial_guess = initial_guess0.copy()
        mask = _get_parameters_mask(including_fit_parameters, lengths)
        residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
        r = residuals.reshape((orm_model).shape)
        t2 = np.zeros([len(initial_guess[mask]), 1])
        for i in range(len(initial_guess[mask])):
            t2[i] = np.sum(np.dot(np.dot(J[i], weights), r.T))

        t3 = (np.dot(Jt, t2)).reshape(-1)
        initial_guess1 = initial_guess0 + t3
        t4 = abs(initial_guess1 - initial_guess0)

        if max(t4) <= eps:
            break
        initial_guess0 = initial_guess1


    return initial_guess0



def loco_correction_ng(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, s_cut, weights=1):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
    r = residuals.reshape(orm_model.shape)

    t2 = np.zeros([len(initial_guess), 1])
    for i in range(len(initial_guess)):
        t2[i] = np.sum(np.dot(np.dot(J[i], weights), r.T))

    results = get_inverse(J, t2, s_cut, weights)
    return results

def loco_correction_ng_old(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, s_cut, weights=1,max_iterations = 10, eps = 1e-6 ):
    Iter = 0
    while True:
        Iter += 1

        if max_iterations is not None and Iter > max_iterations:
            break

        model = orm_model
        len_quads = lengths[0]
        len_corr = lengths[1]
        len_bpm = lengths[2]

        if 'quads' in including_fit_parameters:
            delta_g = initial_guess0[:len_quads]
            J1 = J[:len_quads]
            B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
            model += B

        if 'cor' in including_fit_parameters:
            delta_x = initial_guess0[len_quads:len_quads + len_corr]
            J2 = J[len_quads:len_quads + len_corr]
            # Co = orbit_response_matrix_model * delta_x
            Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
            model += Co

        if 'bpm' in including_fit_parameters:
            delta_y = initial_guess0[len_quads + len_corr:]
            J3 = J[len_quads + len_corr:]
            #G = orbit_response_matrix_model * delta_y[:, np.newaxis]
            G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)

            model += G

        r = orm_measured - model


        t2 = np.zeros([len(initial_guess0), 1])
        for i in range(len(initial_guess0)):
            t2[i] = np.sum(np.dot(np.dot(J[i],weights), r.T)) #############

        t3 = (np.dot(Jt, t2)).reshape(-1)
        initial_guess1 = initial_guess0 + t3
        t4 = abs(initial_guess1 - initial_guess0)

        if max(t4) <= eps:
            break
        initial_guess0 = initial_guess1

    return initial_guess1

def loco_correction(objective1, initial_guess0, C_model, C_measure, J, Jt, lengths, method='lm', eps=1.e-2, weights= 1,  max_iterations=None, verbose=2):
    import numpy as np
    from scipy.optimize import least_squares
    from sklearn.metrics import r2_score, mean_squared_error

    if method == 'lm':
        result = least_squares(objective1, initial_guess0, method=method, verbose=verbose)
        return result.x
    else:
        if method == 'ng':
            Iter = 0

            while True:
                Iter += 1

                if max_iterations is not None and Iter > max_iterations:
                    break

                len_quads = lengths[0]
                len_corr = lengths[1]
                len_bpm = lengths[2]

                delta_g = initial_guess0[:len_quads]
                delta_x = initial_guess0[len_quads:len_quads + len_corr]
                delta_y = initial_guess0[len_quads + len_corr:]

                B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
                Co = C_model * delta_x[:, np.newaxis]
                G = C_model * delta_y

                model = C_model + B + Co + G
                r = C_measure - model

                t2 = np.zeros([len(initial_guess0), 1])
                for i in range(len(initial_guess0)):
                    t2[i] =  np.sum(np.dot(np.dot(J[i], weights), r.T))

                t3 = (np.dot(Jt, t2)).reshape(-1)
                initial_guess1 = initial_guess0 + t3
                t4 = abs(initial_guess1 - initial_guess0)

                if max(t4) <= eps:
                    break
                initial_guess0 = initial_guess1


            return initial_guess0



def objective1(delta_params, C_model, C_measure, J, lengths):

    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    delta_g = delta_params[:len_quads]
    delta_x = delta_params[len_quads:len_quads + len_corr]
    delta_y = delta_params[len_quads + len_corr:]

    D = C_measure - C_model
    B  = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
    Co = C_model * delta_x[:, np.newaxis]
    G = C_model * delta_y

    # Define the objective function to be minimized

    residuals = np.square(D - B - Co - G)

    #residuals = np.square(D - B - Co - G) +  np.square(delta_g) * 0.1



    return residuals.ravel()



def objective(masked_params, orm_residuals, masked_jacobian, weights):
    return np.dot((orm_residuals - np.einsum("ijk,i->jk", masked_jacobian, masked_params)),
                  np.sqrt(weights)).ravel()

def _get_parameters_mask(including_fit_parameters, lengths):
    len_quads, len_corr, len_bpm = lengths
    mask = np.zeros(len_quads + len_corr + len_bpm, dtype=bool)
    mask[:len_quads] = 'quads' in including_fit_parameters
    mask[len_quads:len_quads + len_corr] = 'cor' in including_fit_parameters
    mask[len_quads + len_corr:] = 'bpm' in including_fit_parameters
    return mask


def loco_correction_ng_new(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, s_cut,
                       weights=1, max_iterations=10, tolerance=1e-6):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
    r = residuals.reshape(orm_model.shape)

    t2 = np.zeros([len(initial_guess), 1])
    for i in range(len(initial_guess)):
        t2[i] = np.sum(np.dot(np.dot(J[i], weights), r.T))

    prev_params = initial_guess[mask]
    for _ in range(max_iterations):
        results = get_inverse(J, t2, s_cut, weights)
        new_params = results  # Adjust based on how results are returned from get_inverse

        # Check convergence
        if np.all(np.abs(new_params - prev_params) < tolerance):
            break

        prev_params = new_params

    return new_params


def setCorrection_(ring, quads_strength ,quadInd, add=True, skew=False, family=True):

    if family ==True:

       for fam in range(len(quadInd)):

           q_family = quadInd[fam]

           l = ring[q_family[0]].Length

           len_fam = len(q_family)
           quads_strength[fam] = quads_strength[fam] / len_fam

           for quad in q_family:

               if skew == True:
                    if add == True:
                        ring[quad].PolynomA[1] += -quads_strength[fam]
                    else:
                        ring[quad].PolynomA[1] -= -quads_strength[fam]

               else:
                    if add == True:
                        ring[quad].K += -quads_strength[fam]
                    else:
                        ring[quad].K -= -quads_strength[fam]


    else:

        if skew ==True:
            if add == True:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    ring[qInd].PolynomA[1]  += -quads_strength[i]
            else:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    ring[qInd].PolynomA[1]  -= -quads_strength[i]

        else:
            if add==True:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    ring[qInd].K += -quads_strength[i]
            else:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    ring[qInd].K -= -quads_strength[i]


def setCorrection(ring, quads_strength ,quadInd, add=True, skew=False, family=True):

    if family ==True:

       for fam in range(len(quadInd)):

           q_family = quadInd[fam]

           l = ring[q_family[0]].Length

           for quad in q_family:

               if skew == True:
                    if add == True:
                        ring[quad].PolynomA[1] += -quads_strength[fam] #/l
                    else:
                        ring[quad].PolynomA[1] -= -quads_strength[fam] #/l

               else:
                    if add == True:
                        ring[quad].K += -quads_strength[fam] #/l
                    else:
                        ring[quad].K -= -quads_strength[fam] #/l


    else:



        if skew ==True:
            if add == True:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    l = ring[qInd].Length
                    ring[qInd].PolynomA[1]  += -quads_strength[i] #/l
            else:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    qInd = quadInd[i]
                    l = ring[qInd].Length
                    ring[qInd].PolynomA[1]  -= -quads_strength[i] #/l

        else:
            if add==True:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    qInd = quadInd[i]
                    l = ring[qInd].Length
                    ring[qInd].K += -quads_strength[i] #/l
            else:
                for i in range(len(quadInd)):
                    qInd = quadInd[i]
                    qInd = quadInd[i]
                    l = ring[qInd].Length
                    ring[qInd].K -= -quads_strength[i] #/l



def plot_fit_parameters(ring, ring0,  quadInd, fit_parameters, lengths, including_fit_parameters, hcm_indices, vcm_indices,
                        bpm_x_indices, bpm_y_indices):
    quads_strength = fit_parameters[:lengths[0]]
    cor_cal = fit_parameters[lengths[0]:lengths[0] + lengths[1]]
    bpm_cal = fit_parameters[lengths[0] + lengths[1]:]
    flattened_indices_with_strengths = []

    for subset, strength in zip(quadInd, quads_strength):
        for index in subset:
            flattened_indices_with_strengths.append((index, strength))

    indices, strengths = zip(*flattened_indices_with_strengths)

    sorted_indices_with_strengths = sorted(flattened_indices_with_strengths, key=lambda x: x[0])
    sorted_indices, sorted_strengths = zip(*sorted_indices_with_strengths)

    _, _, twiss = at.get_optics(ring, sorted_indices)
    # s_pos = twiss.s_pos
    s_pos1 = []
    i_ = []
    for i in indices:
        i_.append(i)
        _, _, twiss = at.get_optics(ring, i)
        s_pos_ = twiss.s_pos
        s_pos1.append(s_pos_)

    k_ = [ring0[i].K for i in indices]
    n_q = len(strengths)
    n_q_indices = np.arange(1, n_q + 1)
    if 'quads' in including_fit_parameters:
        def plot_with_unique_colors(x, y, xlabel, ylabel, ax, use_color=True):
            if use_color:
                grouped_indices = {}
                for idx, value in enumerate(y):
                    if value not in grouped_indices:
                        grouped_indices[value] = []
                    grouped_indices[value].append(idx)

                colors = plt.cm.rainbow(np.linspace(0, 1, len(grouped_indices)))

                for value, idxs in grouped_indices.items():
                    ax.scatter([x[i] for i in idxs], [value] * len(idxs),
                               color=colors[list(grouped_indices.keys()).index(value)], label=f'{ylabel}: {value}')
            else:
                ax.scatter(x, y)

            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True, which='both', linestyle=':', color='gray')

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Quadrupoles Index vs Strengths
        plot_with_unique_colors(n_q_indices, strengths, "Quadrupoles Index", r"$\Delta k / k$", axs[0, 0])
        plot_with_unique_colors(s_pos1, strengths, "S_pos[m]", r"$\Delta k / k$", axs[0, 1])
        plot_with_unique_colors(n_q_indices, k_, "Quadrupoles Index", "k", axs[1, 0], use_color=False)
        plot_with_unique_colors(s_pos1, k_, "S_pos[m]", "k", axs[1, 1], use_color=False)

        handles, labels = axs[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    # Plot corrector cal
    if 'cor' in including_fit_parameters:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        n_hcm = len(hcm_indices)  # Assuming hcm_indices is the number of horizontal correctors
        n_vcm = len(vcm_indices)  # Assuming vcm_indices is the number of vertical correctors

        # Generate index arrays for plotting
        hcm_plot_indices = np.arange(1, n_hcm + 1)
        vcm_plot_indices = np.arange(1, n_vcm + 1)

        # Horizontal Correctors
        axs[0].plot(hcm_plot_indices, cor_cal[:n_hcm], '-o', label='Horizontal Correctors', color='tab:blue')
        axs[0].set_title("Horizontal Correctors")
        axs[0].set_xlabel("Corrector Index", fontsize=14)
        axs[0].set_ylabel('Correction Calibration', fontsize=14)
        axs[0].grid(True, which='both', linestyle=':', color='gray')

        # Vertical Correctors
        axs[1].plot(vcm_plot_indices, cor_cal[n_hcm:n_hcm+n_vcm], '-o', label='Vertical Correctors', color='tab:red')
        axs[1].set_title("Vertical Correctors")
        axs[1].set_xlabel("Corrector Index", fontsize=14)
        axs[1].grid(True, which='both', linestyle=':', color='gray')

        plt.tight_layout()
        plt.show()

    # Plot BPM strengths
    if 'bpm' in including_fit_parameters:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        n_bpm_x = len(bpm_x_indices)  # Number of horizontal BPMs
        n_bpm_y = len(bpm_y_indices)  # Number of vertical BPMs

        # Generate index arrays for plotting
        bpm_x_plot_indices = np.arange(1, n_bpm_x + 1)
        bpm_y_plot_indices = np.arange(1, n_bpm_y + 1)

        # Horizontal BPMs
        axs[0].plot(bpm_x_plot_indices, bpm_cal[:n_bpm_x], '-^', label='Horizontal BPMs', color='tab:green')
        axs[0].set_title("Horizontal BPMs")
        axs[0].set_xlabel("BPM Index", fontsize=14)
        axs[0].set_ylabel('BPM Calibration', fontsize=14)
        axs[0].grid(True, which='both', linestyle=':', color='gray')

        # Vertical BPMs
        axs[1].plot(bpm_y_plot_indices, bpm_cal[n_bpm_x:n_bpm_x+n_bpm_y], '-^', label='Vertical BPMs', color='tab:orange')
        axs[1].set_title("Vertical BPMs")
        axs[1].set_xlabel("BPM Index", fontsize=14)
        axs[1].grid(True, which='both', linestyle=':', color='gray')

        plt.tight_layout()
        plt.show()


def plot_fit_parameters_(ring, quadInd, fit_parameters, lengths, including_fit_parameters, hcm_indices, vcm_indices,
                        bpm_x_indices, bpm_y_indices):
    quads_strength = fit_parameters[:lengths[0]]
    cor_cal = fit_parameters[lengths[0]:lengths[0] + lengths[1]]
    bpm_cal = fit_parameters[lengths[0] + lengths[1]:]

    _, _, twiss = at.get_optics(ring, quadInd)
    s_pos = twiss.s_pos

    # Plot quadrupole strengths
    if 'quads' in including_fit_parameters:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.scatter(s_pos, quads_strength, label='Relative quadrupole strength', color='tab:gray', marker='x')
        ax.plot(s_pos, quads_strength, linestyle='-', color='tab:gray', linewidth=1)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\Delta k / k$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        ax.legend(loc="upper left", fontsize=10, frameon=True)
        plt.show()

    # Plot corrector cal
    if 'cor' in including_fit_parameters:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        n_hcm = len(hcm_indices)  # Assuming hcm_indices is the number of horizontal correctors
        n_vcm = len(vcm_indices)  # Assuming vcm_indices is the number of vertical correctors

        # Generate index arrays for plotting
        hcm_plot_indices = np.arange(1, n_hcm + 1)
        vcm_plot_indices = np.arange(1, n_vcm + 1)

        # Horizontal Correctors
        axs[0].plot(hcm_plot_indices, cor_cal[:n_hcm], '-o', label='Horizontal Correctors', color='tab:blue')
        axs[0].set_title("Horizontal Correctors")
        axs[0].set_xlabel("Corrector Index", fontsize=14)
        axs[0].set_ylabel('Correction Calibration', fontsize=14)
        axs[0].grid(True, which='both', linestyle=':', color='gray')

        # Vertical Correctors
        axs[1].plot(vcm_plot_indices, cor_cal[n_hcm:n_hcm+n_vcm], '-o', label='Vertical Correctors', color='tab:red')
        axs[1].set_title("Vertical Correctors")
        axs[1].set_xlabel("Corrector Index", fontsize=14)
        axs[1].grid(True, which='both', linestyle=':', color='gray')

        plt.tight_layout()
        plt.show()

    # Plot BPM strengths
    if 'bpm' in including_fit_parameters:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        n_bpm_x = len(bpm_x_indices)  # Number of horizontal BPMs
        n_bpm_y = len(bpm_y_indices)  # Number of vertical BPMs

        # Generate index arrays for plotting
        bpm_x_plot_indices = np.arange(1, n_bpm_x + 1)
        bpm_y_plot_indices = np.arange(1, n_bpm_y + 1)

        # Horizontal BPMs
        axs[0].plot(bpm_x_plot_indices, bpm_cal[:n_bpm_x], '-^', label='Horizontal BPMs', color='tab:green')
        axs[0].set_title("Horizontal BPMs")
        axs[0].set_xlabel("BPM Index", fontsize=14)
        axs[0].set_ylabel('BPM Calibration', fontsize=14)
        axs[0].grid(True, which='both', linestyle=':', color='gray')

        # Vertical BPMs
        axs[1].plot(bpm_y_plot_indices, bpm_cal[n_bpm_x:n_bpm_x+n_bpm_y], '-^', label='Vertical BPMs', color='tab:orange')
        axs[1].set_title("Vertical BPMs")
        axs[1].set_xlabel("BPM Index", fontsize=14)
        axs[1].grid(True, which='both', linestyle=':', color='gray')

        plt.tight_layout()
        plt.show()


def model_beta_beat(ring, twiss, elements_indexes, plot=False):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    bx = np.array(twiss_error.beta[:, 0] / twiss.beta[:, 0] - 1)
    by = np.array(twiss_error.beta[:, 1] / twiss.beta[:, 1] - 1)
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if plot:
        init_font = plt.rcParams["font.size"]
        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(nrows=2, sharex="all")
        betas = [bx*100, by*100]
        letters = ("x", "y")
        for i in range(2):
            ax[i].plot(s_pos, betas[i])
            ax[i].set_xlabel("s_pos [m]")
            ax[i].set_ylabel(rf'$\Delta\beta_{letters[i]}$ / $\beta_{letters[i]}$ %')
            ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[i].grid(True, which='both', linestyle=':', color='gray')

        fig.show()
        plt.rcParams.update({'font.size': init_font})

    return bx_rms, by_rms


def model_beta_beat(ring, twiss, elements_indexes, plot=False):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    bx2 = np.array(twiss_error.beta[:, 0] / twiss.beta[:, 0])
    by2 = np.array(twiss_error.beta[:, 1] / twiss.beta[:, 1])
    bx = np.array(twiss_error.beta[:, 0] / twiss.beta[:, 0] - 1)
    by = np.array(twiss_error.beta[:, 1] / twiss.beta[:, 1] - 1)
    bx_rms = np.sqrt(np.mean(bx ** 2)) #*100
    by_rms = np.sqrt(np.mean(by ** 2)) #*100

    if plot:
        init_font = plt.rcParams["font.size"]
        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey="all")
        betas = [bx2, by2]
        letters = ("x", "y")
        for i in range(2):
            ax[i].plot(s_pos, betas[i])
            ax[i].set_xlabel("s_pos [m]")
            ax[i].set_ylabel(rf'$\beta_{letters[i]}M$ / $\beta_{letters[i]}T$ ')
            ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[i].grid(True, which='both', linestyle=':', color='gray')

        fig.show()
        plt.rcParams.update({'font.size': init_font})

    return bx_rms, by_rms



def select_equally_spaced_elements(total_elements, num_elements):
    step = len(total_elements) // (num_elements - 1)
    return total_elements[::step]


def get_inverse(J, B, s_cut, weights, plot=False):


    n_resp_mats = len(J)
    sum_corr = np.sum(J, axis=1)
    matrix = np.dot(np.dot(sum_corr, weights), sum_corr.T)

    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    smat = 0.0 * matrix
    si = s ** -1
    sCut = s_cut  # Cut off
    si[sCut:] *= 0.0
    Nk = len(matrix)
    smat[:Nk, :Nk] = np.diag(si)
    inv_matrix = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
    results = np.ravel(np.dot(inv_matrix, B))
    # e = np.ravel(np.dot(matrix, results)) - np.ravel(B)
    return results

def pinv(matrix, num_removed: int = 0, alpha: float = 0, damping: float = 1, plot: bool = False):
    """
    Computes the pseudo-inverse of a matrix using the Singular Value Decomposition (SVD) method.

    Parameters
    ----------
    matrix : ndarray
        The matrix to be inverted.
    num_removed : int, optional
        The number of singular values to be removed from the matrix.
    alpha : float, optional
        The regularization parameter.
    damping : float, optional
        The damping factor.
    plot : bool, optional
        If True, plots the singular values and the damping factor.

    Returns
    -------
    matrix_inv : ndarray
        The pseudo-inverse of the matrix.
    """
    u_mat, s_mat, vt_mat = np.linalg.svd(matrix, full_matrices=False)
    num_singular_values = s_mat.shape[0] - num_removed if num_removed > 0 else s_mat.shape[0]
    available = np.sum(s_mat > 0.)
    keep = min(num_singular_values, available)
    d_mat = np.zeros(s_mat.shape)
    d_mat[:available] = s_mat[:available] / (np.square(s_mat[:available]) + alpha**2) if alpha else 1/s_mat[:available]
    d_mat = damping * d_mat
    matrix_inv = np.dot(np.dot(np.transpose(vt_mat[:keep, :]), np.diag(d_mat[:keep])), np.transpose(u_mat[:, :keep]))
    if plot:
        _plot_singular_values(s_mat, d_mat)
    return matrix_inv

def _plot_singular_values(s_mat, d_mat):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=100, facecolor="w")
    ax[0].semilogy(np.diag(s_mat) / np.max(np.diag(s_mat)), 'o--')
    ax[0].set_xlabel('Number of SV')
    ax[0].set_ylabel('$\sigma/\sigma_0$')
    ax[1].plot(s_mat * d_mat, 'o--')
    ax[1].set_xlabel('Number of SV')
    ax[1].set_ylabel('$\sigma * \sigma^+$')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    fig.show()






def get_inverse2(jacobian, B, s_cut, weights, showPlots=False):
    n_resp_mats = len(jacobian)
    # matrix = np.zeros([n_resp_mats, n_resp_mats])
    # for i in range(n_resp_mats):
    #    for j in range(n_resp_mats):
    #        matrix[i, j] = np.sum(np.dot(np.dot(jacobian[i], weights), jacobian[j].T))
    sum_ = np.sum(jacobian, axis=1)  # Sum over i and j for all planes
    matrix = sum_ @ weights @ sum_.T
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    smat = 0.0 * matrix
    si = s ** -1
    n_sv = s_cut
    si[n_sv:] *= 0.0
    smat[:n_resp_mats, :n_resp_mats] = np.diag(si)
    matrixi = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))


    r = (np.dot(matrixi, B)).reshape(-1)
    #e = np.dot(matrix, r).reshape(-1) - B.reshape(-1)

    return r



def simulateShiftErrors(lattice, shiftx,shifty, elementsInd=None, sigmaCut=None, relative=False):
    for i in elementsInd:
        a =  generateTrucatedGaussian(mean=0, sigma=shiftx, cutValue=sigmaCut*shiftx)
        b =  generateTrucatedGaussian(mean=0, sigma=shifty, cutValue=sigmaCut*shifty)
        at.shift_elem(lattice[i],a, b,
                      relative)


def simulateTilttErrors(lattice, tilts,pitches, yaws, elementsInd=None, sigmaCut=None, relative=True):
    for i in elementsInd:
        a = generateTrucatedGaussian(mean=0, sigma=tilts, cutValue=sigmaCut*tilts)
        b = generateTrucatedGaussian(mean=0, sigma=pitches, cutValue=sigmaCut*pitches)
        c = generateTrucatedGaussian(mean=0, sigma=yaws, cutValue=sigmaCut*yaws)
        #at.tilt_elem(lattice[i], rots=a, relative=relative)
        at.rotate_elem(lattice[i], tilt=a,
        pitch=b, yaw=c, relative=relative)

def generateTrucatedGaussian(mean, sigma, cutValue):
    seed(5)
    numberFound = False
    while(numberFound == False):
        a = (mean + sigma * np.random.randn())
        if - cutValue > a or a > cutValue:
            numberFound = False
        else:
            numberFound = True
            return a

def simulateFieldErrors(lattice, gradErr, elementsInd, print_err):
    quads_strength = []
    for i in elementsInd:
        quads_strength.append(lattice[i].K)

    for i in elementsInd:
        seed(3)
        a = (1 + gradErr * np.random.randn())
        lattice[i].K *= a

    if print_err ==  True:
        quads_strength_err =[]
        for i in elementsInd:
            quads_strength_err.append(lattice[i].K)
        quads_err = [a - b for a, b in zip(quads_strength_err, quads_strength)]

        vectors = [quads_strength, quads_strength_err, quads_err]
        vector_names = ['nominal_quads_strength', 'err_quads_strength', 'error value']
        data = {name: vector for name, vector in zip(vector_names, vectors)}
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))
        print(df)


    return  quads_err



def rms_orbits(ring, elements_indexes, makeplot):
    _, _, elemdata = at.get_optics(ring, elements_indexes)
    closed_orbitx = elemdata.closed_orbit[:, 0]
    closed_orbity = elemdata.closed_orbit[:, 2]
    s_pos = elemdata.s_pos

    if makeplot == True:
        import matplotlib.pyplot as plt

        plt.rc('font', size=13)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # Adjust the figsize to your preference

        ax1.plot(s_pos, closed_orbitx / 1.e-06)
        ax1.set_xlabel("s_pos [m]", fontsize=12)  # Adjust the fontsize as needed
        ax1.set_ylabel(r"closed_orbit x [$\mu$m]", fontsize=12)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax1.grid(True, which='both', linestyle=':', color='gray')
        ax1.set_title("Closed orbit x")


        ax2.plot(s_pos, closed_orbity / 1.e-06)
        ax2.set_xlabel("s_pos [m]", fontsize=12)  # Adjust the fontsize as needed
        ax2.set_ylabel(r"closed_orbit y [$\mu$m]", fontsize=12)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax2.grid(True, which='both', linestyle=':', color='gray')
        ax2.set_title("Closed orbit y")


        plt.tight_layout()

        plt.show()

    rmsx =np.sqrt(np.mean(closed_orbitx ** 2))
    rmsy =np.sqrt(np.mean(closed_orbity ** 2))

    return rmsx, rmsy

def getDispersionErr(ring, twiss, elements_indexes, makeplot):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    Beta_x = twiss_error.beta[:, 0]
    Beta_y = twiss_error.beta[:, 1]
    bx = np.array((twiss_error.dispersion[:, 0] - twiss.dispersion[:, 0]))
    by = np.array((twiss_error.dispersion[:, 2] - twiss.dispersion[:, 2]))
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if makeplot == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_x)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\eta_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal dispersion')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_y)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\eta_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical dispersion')
        plt.show()

    return bx_rms*1000, by_rms*1000


def corrections_plots(ring ,twiss, twiss_err, plot):
    s_pos_err = twiss_err.s_pos
    Beta_x_err = twiss_err.beta[:, 0]
    Beta_y_err = twiss_err.beta[:, 1]
    closed_orbitx_err = twiss_err.closed_orbit[:, 0]
    closed_orbity_err = twiss_err.closed_orbit[:, 2]
    elements_indexes = at.get_refpts(ring,at.elements.Monitor)

    _, _, twiss_cor = at.get_optics(ring, elements_indexes)
    s_pos = twiss_cor.s_pos
    Beta_x = twiss_cor.beta[:, 0]
    Beta_y = twiss_cor.beta[:, 1]
    closed_orbitx = twiss_cor.closed_orbit[:, 0]
    closed_orbity = twiss_cor.closed_orbit[:, 2]

    if plot == 'orbit':
        fig, ax = plt.subplots()
        ax.plot(s_pos, closed_orbitx_err / 1.e-06, label="Closed orbit with errors")
        ax.plot(s_pos, closed_orbitx / 1.e-06, label="Closed orbit after correction")
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"Horizontal closed_orbit x [$\mu$m]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        ax.legend()
        plt.title("Horizontal Closed orbit")
        plt.show()

        fig, ax = plt.subplots()

        ax.plot(s_pos, closed_orbity_err / 1.e-06, label="Closed orbit with errors")
        ax.plot(s_pos, closed_orbity / 1.e-06, label="Closed orbit after correction")
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"Vertical closed_orbit x [$\mu$m]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        ax.legend()
        plt.title("Vertical Closed orbit")
        plt.show()
    if plot == 'beta':
        bx_err = np.array((twiss_err.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0])
        by_err = np.array((twiss_err.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1])
        bx = np.array((twiss_cor.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0])
        by = np.array((twiss_cor.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1])

        fig = plt.figure()
        plt.plot(twiss.s_pos, bx_err, label='before correction')
        plt.plot(twiss.s_pos, bx, label='after correction')
        plt.xlabel('s[m]')
        plt.ylabel(r'$\frac{\Delta \beta_x}{\beta_x}%$')
        plt.title("Horizontal Beta beating")
        plt.legend(loc='upper left')
        fig = plt.figure()
        plt.plot(twiss.s_pos, by_err, label='before correction')
        plt.plot(twiss.s_pos, by, label='after correction')
        plt.xlabel('s[m]')
        plt.ylabel(r'$\frac{\Delta \beta_y}{\beta_y}%$')
        plt.title("Vertical Beta beating")
        plt.legend(loc='upper left')
        plt.show()


def load_names(filename):
    with open(filename, 'r') as file:
        names = [line.strip() for line in file.readlines()]
    return names


def get_my_orbit(closed_orbit, bpm_calibration,bpm_noise):
    closed_orbit_err = [co * (1 + cal) + noise for co, cal, noise in zip(closed_orbit, bpm_calibration, bpm_noise)]
    return closed_orbit_err

def GenerateCalrr(used_bpm1,sigma1, bpmx,  used_bpm2 , sigma2, bpmy, used_correctors1, sigma3,Corx , used_correctors2, sigma4, Cory, Hor_bpm_noise1, sigma5, Ver_bpm_noise1, sigma6, print_output ):

   if bpmx == True:
        bpmx_calibration = []
        for i, sigma in zip(used_bpm1, sigma1):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            bpmx_calibration.append(rand_numb)
   else:
       bpmx_calibration = []
       for i in used_bpm1:
           bpmx_calibration.append(0)

   if bpmy == True:


       bpmy_calibration = []

       for i, sigma in zip(used_bpm2, sigma2):
           rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
           bpmy_calibration.append(rand_numb)

   else:
       bpmy_calibration = []
       for i in used_bpm2:
           bpmy_calibration.append(0)

   if Corx == True:
        corx_calibration = []
        for i, sigma in zip(used_correctors1, sigma3):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            corx_calibration.append(rand_numb)

   else:
        corx_calibration = []
        for i in used_correctors1:
            corx_calibration.append(0)

   if Cory == True:
        cory_calibration = []
        for i, sigma in zip(used_correctors2, sigma4):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            cory_calibration.append(rand_numb)

   else:
    cory_calibration = []
    for i in used_correctors2:
        cory_calibration.append(0)

   if Hor_bpm_noise1 == True:
        Hor_bpm_noise = []
        for i, sigma in zip(used_bpm1, sigma5):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            Hor_bpm_noise.append(rand_numb)
   else:
        Hor_bpm_noise = []
        for i in used_bpm1:
            Hor_bpm_noise.append(0)

   if Ver_bpm_noise1 == True:
        Ver_bpm_noise = []
        for i, sigma in zip(used_bpm2, sigma6):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            Ver_bpm_noise.append(rand_numb)
   else:
        Ver_bpm_noise = []
        for i in used_bpm2:
            Ver_bpm_noise.append(0)

   vectors = [bpmx_calibration, bpmy_calibration, corx_calibration, cory_calibration, Hor_bpm_noise, Ver_bpm_noise]
   vector_names = ['bpmx_calibration', 'bpmy_calibration', 'corx_calibration', 'cory_calibration', 'Hor_bpm_noise',
                   'Ver_bpm_noise']
   data = {name: vector for name, vector in zip(vector_names, vectors)}
   df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))

   if print_output == True:
      print(df)

   return bpmx_calibration,  bpmy_calibration,  corx_calibration, cory_calibration, Hor_bpm_noise,Ver_bpm_noise, df


def measured_orm(dkick, ring, used_correctors_ind1,used_correctors_ind2, used_bpm1,used_bpm2, corCal1, corCal2, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey, includeDispersion=False, include_coupling=True):


    Cx0, Cxy0 = ORM_x_G(dkick, ring, used_correctors_ind1, used_bpm1,used_bpm2, corCal1, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey,includeDispersion)
    Cy0, Cyx0 = ORM_y_G(dkick, ring, used_correctors_ind2,  used_bpm1,used_bpm2,corCal2, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey,includeDispersion)

    if include_coupling:
        Y = np.zeros((Cx0.shape[0] + Cy0.shape[0], Cx0.shape[1] + Cy0.shape[1]))
        Y[:Cx0.shape[0], :Cx0.shape[1]] = Cx0
        Y[:Cx0.shape[0], Cx0.shape[1]:] = Cxy0
        Y[Cx0.shape[0]:, :Cx0.shape[1]] = Cyx0
        Y[Cx0.shape[0]:, Cx0.shape[1]:] = Cy0
    else:
        # When coupling is not included, return only the diagonal blocks (Cx0, Cy0), ignoring coupling terms
        # Note: The output shape changes, it becomes a block diagonal matrix with zeroed coupling terms
        Y = np.zeros((Cx0.shape[0] + Cy0.shape[0], Cx0.shape[1] + Cy0.shape[1]))
        Y[:Cx0.shape[0], :Cx0.shape[1]] = Cx0
        Y[Cx0.shape[0]:, Cx0.shape[1]:] = Cy0


    return Y


def ORM_x_G(dkick, ring, used_correctors_ind, used_bpm1,used_bpm2, corCal, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey,includeDispersion=False):
    cxx_p = []
    cxy_p = []
    cxx_m = []
    cxy_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm1)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    _, _, elemdata = at.get_optics(ring, used_bpm2)
    closed_orbity0 = elemdata.closed_orbit[:, 2]

    #closed_orbitx0 = get_my_orbit(closed_orbitx0, bpm_calibrationx, bpm_noisex) #
    #closed_orbity0 = get_my_orbit(closed_orbity0, bpm_calibrationy, bpm_noisey) #


    for i in range(len(used_correctors_ind)):


        a = ring[used_correctors_ind[i]].KickAngle[0]

        ring[used_correctors_ind[i]].KickAngle[0] = ((dkick/2)*(1+corCal[i])) + a
        _, _, elemdata = at.get_optics(ring, used_bpm1)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        _, _, elemdata = at.get_optics(ring, used_bpm2)
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0

        closed_orbitx = get_my_orbit(closed_orbitx, bpm_calibrationx , bpm_noisex)
        closed_orbity = get_my_orbit(closed_orbity, bpm_calibrationy,bpm_noisey)

        cxx_p.append(closed_orbitx)
        cxy_p.append(closed_orbity)
        ring[used_correctors_ind[i]].KickAngle[0] = -((dkick/2)*(1+corCal[i])) + a

        _, _, elemdata = at.get_optics(ring, used_bpm1)

        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        _, _, elemdata = at.get_optics(ring, used_bpm2)
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        closed_orbitx = get_my_orbit(closed_orbitx, bpm_calibrationx,bpm_noisex)
        closed_orbity = get_my_orbit(closed_orbity, bpm_calibrationy,bpm_noisey)
        cxx_m.append(closed_orbitx)
        cxy_m.append(closed_orbity)
        ring[used_correctors_ind[i]].KickAngle[0] = a
    Cxx = (np.squeeze(cxx_p)-np.squeeze(cxx_m))
    Cxy = (np.squeeze(cxy_p)-np.squeeze(cxy_m))

    if includeDispersion==True:

        lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm1)
        Eta_xx = lindata['dispersion'][:, 0]
        Eta_yy = lindata['dispersion'][:, 2]

        Cxx = np.vstack((Cxx, Eta_xx))
        Cxy = np.vstack((Cxy, Eta_yy))

    Cxx = Cxx / dkick
    Cxy = Cxy / dkick

    return (Cxx), (Cxy)


def ORM_y_G(dkick, ring, used_correctors_ind,  used_bpm1,used_bpm2,corCal, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey,includeDispersion=False):
    cyy_p = []
    cyx_p = []
    cyy_m = []
    cyx_m = []
    cal_erry =[]

    _, _, elemdata = at.get_optics(ring, used_bpm1)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    _, _, elemdata = at.get_optics(ring, used_bpm2)
    closed_orbity0 = elemdata.closed_orbit[:, 2]


    #closed_orbitx0 = get_my_orbit(closed_orbitx0, bpm_calibrationx, bpm_noisex) #
    #closed_orbity0 = get_my_orbit(closed_orbity0, bpm_calibrationy, bpm_noisey) #

    for i in range(len(used_correctors_ind)):
        a = ring[used_correctors_ind[i]].KickAngle[1]
        ring[used_correctors_ind[i]].KickAngle[1] = ((dkick/2)*(1+corCal[i])) + a
        _, _, elemdata = at.get_optics(ring, used_bpm1)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        _, _, elemdata = at.get_optics(ring, used_bpm2)
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        closed_orbitx = get_my_orbit(closed_orbitx, bpm_calibrationx,bpm_noisex)
        closed_orbity = get_my_orbit(closed_orbity, bpm_calibrationy,bpm_noisey)
        cyy_p.append(closed_orbity)
        cyx_p.append(closed_orbitx)
        ring[used_correctors_ind[i]].KickAngle[1] = -((dkick/2)*(1+corCal[i])) + a
        _, _, elemdata = at.get_optics(ring, used_bpm1)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        _, _, elemdata = at.get_optics(ring, used_bpm2)
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        closed_orbitx = get_my_orbit(closed_orbitx, bpm_calibrationx,bpm_noisex)
        closed_orbity = get_my_orbit(closed_orbity, bpm_calibrationy,bpm_noisey)
        cyy_m.append(closed_orbity)
        cyx_m.append(closed_orbitx)


        ring[used_correctors_ind[i]].KickAngle[1] = a
    Cyy = (np.squeeze(cyy_p)-np.squeeze(cyy_m))
    Cyx = (np.squeeze(cyx_p)-np.squeeze(cyx_m))

    if includeDispersion == True:
        lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm1)
        Eta_xx = lindata['dispersion'][:, 0]
        Eta_yy = lindata['dispersion'][:, 2]

        Cyy = np.vstack((Cyy, Eta_yy))
        Cyx = np.vstack((Cyx, Eta_xx))

    Cyy = Cyy / dkick
    Cyx = Cyx / dkick


    return (Cyy), (Cyx)



def plot_orm(X, Y):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow((X))
    plt.title('Model ORM')
    plt.subplot(1, 2, 2)
    plt.imshow((Y), cmap='cividis')
    plt.title('Measured ORM')
    plt.tight_layout()
    plt.show()

def weight_matrix(ring, Noise_BPMx, Noise_BPMy,show_plot=True ): #Noise_BPMx and Noise_BPMy are in (mm)

    bpm_noisex = Noise_BPMx / 1000
    min_value = np.min(abs(bpm_noisex))
    bpm_noisex /= min_value
    sigmax = bpm_noisex

    bpm_noisey = Noise_BPMy / 1000
    min_value = np.min(abs(bpm_noisey))
    bpm_noisey /= min_value
    sigmay = bpm_noisey
    sigma = np.concatenate((sigmax, sigmay))
    diagonal_values = 1 / np.square(sigma)
    Weight = np.diag(diagonal_values)

    if show_plot:
        _, _, twiss_error = at.get_optics(ring, at.get_refpts(ring, at.elements.Monitor))
        s_pos = twiss_error.s_pos
        fig, ax1 = plt.subplots(figsize=(6, 2))

        ax1.scatter(s_pos, Noise_BPMx, label='Hor. bpms noise', color='tab:blue', marker='x')
        ax1.set_xlabel("s_pos [m]", fontsize=14)
        ax1.set_ylabel('BPMs noise [mm]', fontsize=14)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax1.grid(True, which='both', linestyle=':', color='gray')
        ax1.scatter(s_pos, Noise_BPMy, label='Ver. bpms noise', color='tab:red', marker='x')
        legend = ax1.legend(loc="upper left", fontsize=10, frameon=True)
        plt.show()

    return Weight


def GenerateCalrr(used_bpm1,sigma1, bpmx,  used_bpm2 , sigma2, bpmy, used_correctors1, sigma3,Corx , used_correctors2, sigma4, Cory, Hor_bpm_noise1, sigma5, Ver_bpm_noise1, sigma6, print_output ):

   if bpmx == True:
        bpmx_calibration = []
        for i, sigma in zip(used_bpm1, sigma1):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            bpmx_calibration.append(rand_numb)
   else:
       bpmx_calibration = []
       for i in used_bpm1:
           bpmx_calibration.append(0)

   if bpmy == True:


       bpmy_calibration = []

       for i, sigma in zip(used_bpm2, sigma2):
           rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
           bpmy_calibration.append(rand_numb)

   else:
       bpmy_calibration = []
       for i in used_bpm2:
           bpmy_calibration.append(0)

   if Corx == True:
        corx_calibration = []
        for i, sigma in zip(used_correctors1, sigma3):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            corx_calibration.append(rand_numb)

   else:
        corx_calibration = []
        for i in used_correctors1:
            corx_calibration.append(0)

   if Cory == True:
        cory_calibration = []
        for i, sigma in zip(used_correctors2, sigma4):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            cory_calibration.append(rand_numb)

   else:
    cory_calibration = []
    for i in used_correctors2:
        cory_calibration.append(0)

   if Hor_bpm_noise1 == True:
        Hor_bpm_noise = []
        for i, sigma in zip(used_bpm1, sigma5):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            Hor_bpm_noise.append(rand_numb)
   else:
        Hor_bpm_noise = []
        for i in used_bpm1:
            Hor_bpm_noise.append(0)

   if Ver_bpm_noise1 == True:
        Ver_bpm_noise = []
        for i, sigma in zip(used_bpm2, sigma6):
            rand_numb = generateTrucatedGaussian(mean=0, sigma=sigma, cutValue=2.5 * sigma)
            Ver_bpm_noise.append(rand_numb)
   else:
        Ver_bpm_noise = []
        for i in used_bpm2:
            Ver_bpm_noise.append(0)
   if print_output == True:

       vectors = [bpmx_calibration, bpmy_calibration, corx_calibration, cory_calibration, Hor_bpm_noise, Ver_bpm_noise]
       vector_names = ['bpmx_calibration', 'bpmy_calibration', 'corx_calibration', 'cory_calibration', 'Hor_bpm_noise',
                       'Ver_bpm_noise']
       data = {name: vector for name, vector in zip(vector_names, vectors)}
       df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))
       print(df)

   return bpmx_calibration,  bpmy_calibration,  corx_calibration, cory_calibration, Hor_bpm_noise,Ver_bpm_noise, df

def extract_names_with_q(data):
    if not isinstance(data, str):
        raise ValueError("Data must be a string")

    pattern = r"Q\w+"
    names = re.findall(pattern, str(data))
    return names


def Full_Jacobian(j , C_model, used_cor_ind_h, used_cor_ind_v, bpm_indexes_h, bpm_indexes_v ):
    n_correctors = len(used_cor_ind_h) + len(used_cor_ind_v)
    n_bpms = len(bpm_indexes_h) + len(bpm_indexes_v)
    j_cor = np.zeros((n_correctors,) + C_model.shape)
    for i in range(n_correctors):
        j_cor[i, i, :] = C_model[i, :]  # a single column of response matrix corresponding to a corrector
    j_bpm = np.zeros((n_bpms,) + C_model.shape)
    for i in range(n_bpms):
        j_bpm[i, :, i] = C_model[:, i]  # a single row of response matrix corresponding to a given plane of BPM
    return np.concatenate((j, j_cor, j_bpm), axis=0)



def Remove_orm_coupling(orm, hcm_index, bpm_x_ind):
    orm[len(hcm_index):, :len(bpm_x_ind)] = 0
    orm[:len(hcm_index), len(bpm_x_ind):] = 0

    return orm


def generatingCorResponse_(ring, Cx, Cy, Cxy, Cyx,cor_cal, correctors_kick,used_correctors,used_cor_h,used_cor_v ,used_bpm1,used_bpm2, debug=False):
    dCorx = []
    dCorxy = []
    dCoryx = []
    dCory = []
    for nDim in range(2):
        for cor_ind in range(len(used_correctors[nDim])):
            if debug == True :
               print('generating response to Cor of ord,', nDim, 'index', used_correctors[nDim][cor_ind])
            C1x, C1xy, C1y, C1yx = corSensitivityMatrices_(ring,cor_ind, nDim, cor_cal, correctors_kick,used_cor_h,used_cor_v, used_bpm1,used_bpm2)
            dCorx.append((C1x - Cx) / cor_cal)
            dCory.append((C1y - Cy) / cor_cal)
            dCorxy.append((C1xy - Cxy) / cor_cal)
            dCoryx.append((C1yx - Cyx) / cor_cal)

    j2 = np.zeros(
        (len(used_cor_h) + len(used_cor_v), Cx.shape[0] + Cy.shape[0], Cx.shape[1] + Cy.shape[1]))
    j2[:, :Cx.shape[0], :Cx.shape[1]] = dCorx
    j2[:, :Cx.shape[0], Cx.shape[1]:] = dCorxy
    j2[:, Cx.shape[0]:, :Cx.shape[1]] = dCoryx
    j2[:, Cx.shape[0]:, Cx.shape[1]:] = dCory

    return j2



def corSensitivityMatrices_(ring,cor_ind,nDim, cor_cal, correctors_kick,used_cor_h,used_cor_v, used_bpm1,used_bpm2):
    common_sigma = 0
    if nDim == 0:

        corx_calibration = [common_sigma] * len(used_cor_h)
        cory_calibration = [common_sigma] * len(used_cor_v)
        corx_calibration[cor_ind] = cor_cal

    else:

        corx_calibration = [common_sigma] * len(used_cor_h)
        cory_calibration = [common_sigma] * len(used_cor_v)
        cory_calibration[cor_ind] = cor_cal


    common_sigma = 0
    bpmx_calibration = [common_sigma] * len(used_bpm1)
    bpmy_calibration = [common_sigma] * len(used_bpm2)
    Hor_noise = [common_sigma] * len(used_bpm1)
    Ver_noise = [common_sigma] * len(used_bpm2)


    qxx, qxy = ORM_x_G(correctors_kick, ring, used_cor_h,  used_bpm1,used_bpm2,corx_calibration, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)

    qyy, qyx = ORM_y_G(correctors_kick, ring, used_cor_v,  used_bpm1,used_bpm2,cory_calibration,  bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)


    return  qxx, qxy, qyy, qyx


def generatingBpmrResponse_(ring, Cx, Cy, Cxy, Cyx,correctors_kick,  bpm_cal,used_bpms,used_cor_h,used_cor_v ,used_bpm1,used_bpm2, debug=False):
    dBpmx = []
    dBpmxy = []
    dBpmyx = []
    dBpmy = []
    for nDim in range(2):
        for bpm_ind in range(len(used_bpms[nDim])):
            if debug == True :
               print('generating response to BPM of ord,', nDim, 'index', used_bpms[nDim][bpm_ind])
            C1x, C1xy, C1y, C1yx = bpmSensitivityMatrices_(ring,correctors_kick,bpm_ind, nDim, bpm_cal,used_cor_h,used_cor_v, used_bpm1,used_bpm2)
            dBpmx.append((C1x - Cx) / bpm_cal)
            dBpmxy.append((C1xy - Cxy) / bpm_cal)
            dBpmyx.append((C1yx - Cyx) / bpm_cal)
            dBpmy.append((C1y - Cy) / bpm_cal)

    j3 = np.zeros((len(used_bpm1) + len(used_bpm2), Cx.shape[0] + Cy.shape[0], Cx.shape[1] + Cy.shape[1]))
    j3[:, :Cx.shape[0], :Cx.shape[1]] = dBpmx
    j3[:, :Cx.shape[0], Cx.shape[1]:] = dBpmxy
    j3[:, Cx.shape[0]:, :Cx.shape[1]] = dBpmyx
    j3[:, Cx.shape[0]:, Cx.shape[1]:] = dBpmy

    return j3



def bpmSensitivityMatrices_(ring,correctors_kick, bpm_ind,nDim, bpm_cal,used_cor_h,used_cor_v, used_bpm1,used_bpm2):
    common_sigma = 0
    if nDim == 0:

        bpmx_calibration = [common_sigma] * len(used_bpm1)
        bpmy_calibration = [common_sigma] * len(used_bpm2)
        bpmx_calibration[bpm_ind] = bpm_cal

    else:

        bpmx_calibration = [common_sigma] * len(used_bpm1)
        bpmy_calibration = [common_sigma] * len(used_bpm2)
        bpmy_calibration[bpm_ind] = bpm_cal


    common_sigma = 0
    corx_calibration = [common_sigma] * len(used_cor_h)
    cory_calibration = [common_sigma] * len(used_cor_v)
    Hor_noise = [common_sigma] * len(used_bpm1)
    Ver_noise = [common_sigma] * len(used_bpm2)



    qxx, qxy = ORM_x_G(correctors_kick, ring, used_cor_h,  used_bpm1,used_bpm2,corx_calibration, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)

    qyy, qyx = ORM_y_G(correctors_kick, ring, used_cor_v,  used_bpm1,used_bpm2,cory_calibration, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)


    return  qxx, qxy, qyy, qyx



def loco_correction_ng_new(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, s_cut,
                       weights=1, max_iterations=1000, tolerance=1e-50):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask(including_fit_parameters, lengths)
    residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
    r = residuals.reshape(orm_model.shape)

    t2 = np.zeros([len(initial_guess), 1])
    for i in range(len(initial_guess)):
        t2[i] = np.sum(np.dot(np.dot(J[i], weights), r.T))

    prev_params = initial_guess[mask]
    for _ in range(max_iterations):
        results = get_inverse(J, t2, s_cut, weights)
        new_params = results

        if np.all(np.abs(new_params - prev_params) < tolerance):
            break

        prev_params = new_params

    return new_params


def loco_correction_ng_new2(initial_guess0, orm_model, orm_measured, J, Jt, lengths, including_fit_parameters, s_cut,
                            weights=1, max_iterations=1000, tolerance=1e-10):
    initial_guess = initial_guess0.copy()
    mask = _get_parameters_mask(including_fit_parameters, lengths)

    # Initial residuals calculation
    residuals = objective(initial_guess[mask], orm_measured - orm_model, J[mask, :, :], weights)
    prev_residuals_norm = np.linalg.norm(residuals)
    new_params = initial_guess[mask]

    for iteration in range(max_iterations):
        # Calculate 't2' using the current parameters
        r = residuals.reshape(orm_model.shape)
        t2 = np.zeros(len(new_params))
        for i in range(len(new_params)):
            t2[i] = np.sum(np.dot(np.dot(J[mask][i], weights), r.T))

        # Update parameters using Newton-Gauss method (pseudo-inverse or similar)
        delta_params = get_inverse(J[mask], t2[:, np.newaxis], s_cut, weights)
        new_params += delta_params.squeeze()

        # Calculate new residuals with updated parameters
        residuals = objective(new_params, orm_measured - orm_model, J[mask, :, :], weights)
        residuals_norm = np.linalg.norm(residuals)

        # Check for convergence: if the improvement is smaller than the tolerance, stop.
        if abs(prev_residuals_norm - residuals_norm) < tolerance:
            print(f"Convergence reached due to tolerance after {iteration + 1} iterations.")
            break
        #prev_residuals_norm = residuals_norm

    #if iteration + 1 == max_iterations and abs(prev_residuals_norm - residuals_norm) >= tolerance:
    print(f"end at iterations ({iteration}) the tolerance condition({tolerance})  reached ({prev_residuals_norm - residuals_norm}).")

    # Ensure the final parameters are assigned back correctly, considering the mask
    final_params = initial_guess.copy()
    final_params[mask] = new_params

    return new_params



def loco_correction_ng_old(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, Jt, lengths, including_fit_parameters,W = 1, max_iterations =100, eps= 1e-10):
    Iter = 0
    while True:
        Iter += 1

        if max_iterations is not None and Iter > max_iterations:
            break

        model = orbit_response_matrix_model
        len_quads = lengths[0]
        len_corr = lengths[1]
        len_bpm = lengths[2]

        if 'quads' in including_fit_parameters:
            delta_g = initial_guess0[:len_quads]
            J1 = J[:len_quads]
            B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
            model += B

        if 'cor' in including_fit_parameters:
            delta_x = initial_guess0[len_quads:len_quads + len_corr]
            J2 = J[len_quads:len_quads + len_corr]
            Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
            model += Co

        if 'bpm' in including_fit_parameters:
            delta_y = initial_guess0[len_quads + len_corr:]
            J3 = J[len_quads + len_corr:]
            G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)

            model += G

        r = orbit_response_matrix_measured - model

        t2 = np.zeros([len(initial_guess0), 1])
        for i in range(len(initial_guess0)):
            t2[i] = np.sum(np.dot(np.dot(J[i],W), r.T)) #############

        t3 = (np.dot(Jt, t2)).reshape(-1)
        initial_guess1 = initial_guess0 + t3
        t4 = abs(initial_guess1 - initial_guess0)

        if max(t4) <= eps:
            break
        initial_guess0 = initial_guess1

    return initial_guess0


def loco_correction_ng_old2(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, Jt, lengths, including_fit_parameters,W = 1, max_iterations =100, eps= 1e-10):
    Iter = 0

    model = orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        delta_g = initial_guess0[:len_quads]
        J1 = J[:len_quads]
        B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
        model += B

    if 'cor' in including_fit_parameters:
        delta_x = initial_guess0[len_quads:len_quads + len_corr]
        J2 = J[len_quads:len_quads + len_corr]
        Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
        model += Co

    if 'bpm' in including_fit_parameters:
        delta_y = initial_guess0[len_quads + len_corr:]
        J3 = J[len_quads + len_corr:]
        G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)

        model += G

    r = orbit_response_matrix_measured - model

    t2 = np.zeros([len(initial_guess0), 1])
    for i in range(len(initial_guess0)):
        t2[i] = np.sum(np.dot(np.dot(J[i],W), r.T)) #############

    t3 = (np.dot(Jt, t2)).reshape(-1)

    return t3
