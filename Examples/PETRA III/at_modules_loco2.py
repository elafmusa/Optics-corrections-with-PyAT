from at import *
import at.plot
from pylab import *
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error

def select_equally_spaced_elements(total_elements, num_elements):
    step = len(total_elements) // (num_elements - 1)
    indexes = total_elements[::step]
    return indexes

def ORM_x1(dkick, ring, used_correctors_ind, used_bpm):
    cxx_p = []
    cxy_p = []
    cxx_m = []
    cxy_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        cor_index = cor_index[0]

        a = ring[cor_index].KickAngle[0]
        ring[cor_index].KickAngle[0] = dkick/2 + a
        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cxx_p.append(closed_orbitx)
        cxy_p.append(closed_orbity)

        ring[cor_index].KickAngle[0] = -dkick / 2 + a

        _, _, elemdata = at.get_optics(ring, used_bpm)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        cxx_m.append(closed_orbitx)
        cxy_m.append(closed_orbity)


        ring[cor_index].KickAngle[0] = a
    Cxx = (np.squeeze(cxx_p)-np.squeeze(cxx_m))/dkick
    Cxy = (np.squeeze(cxy_p)-np.squeeze(cxy_m))/dkick

    return (Cxx), (Cxy)


def ORM_y1(dkick, ring, used_correctors_ind, used_bpm):
    cyy_p = []
    cyx_p = []
    cyy_m = []
    cyx_m = []

    _, _, elemdata = at.get_optics(ring, used_bpm)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]
    for cor_index in used_correctors_ind:
        cor_index = cor_index[0]
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
    Cyy = (np.squeeze(cyy_p)-np.squeeze(cyy_m))/dkick
    Cyx = (np.squeeze(cyx_p)-np.squeeze(cyx_m))/dkick

    return (Cyy), (Cyx)


def generatingQuadsResponse_(ring, Cx, Cy, Cxy, Cyx, correctors_kick,used_cor_h,used_cor_v,used_bpm1,used_bpm2, quads_indexes, dk, debug=False):
    dCx = []
    dCy = []
    dCxy = []
    dCyx = []
    for quad_index in quads_indexes:
        if debug == True :
           print('generating response to quad of index', quad_index)
        C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices_1(ring,correctors_kick,used_cor_h,used_cor_v, used_bpm1,used_bpm2, quad_index, dk)
        dCx.append((C1x - Cx) / dk)
        dCy.append((C1y - Cy) / dk)
        dCxy.append((C1xy - Cxy) / dk)
        dCyx.append((C1yx - Cyx) / dk)

    j1 = np.zeros((len(quads_indexes), Cx.shape[0] + Cy.shape[0], Cx.shape[1] + Cy.shape[1]))
    j1[:, :Cx.shape[0], :Cx.shape[1]] = dCx
    j1[:, :Cx.shape[0], Cx.shape[1]:] = dCxy ###########
    j1[:, Cx.shape[0]:, :Cx.shape[1]] = dCyx
    j1[:, Cx.shape[0]:, Cx.shape[1]:] = dCy

    return j1


def quadsSensitivityMatrices_(ring,correctors_kick,used_cor_h,used_cor_v, used_bpm1,used_bpm2, quad_index, dk):
    strength_before = ring[quad_index].K
    ring[quad_index].K = strength_before + dk
    qxx, qxy = ORM_x1(correctors_kick, ring, used_cor_h, used_bpm1)
    qyy, qyx = ORM_y1(correctors_kick, ring, used_cor_v,  used_bpm2)

    ring[quad_index].K = strength_before
    return  qxx, qxy, qyy, qyx


def quadsSensitivityMatrices_1(ring,correctors_kick,used_cor_h,used_cor_v, used_bpm1,used_bpm2, quad_index, dk):

    i = quad_index

    strength_before = ring[i].K
    l = ring[i].Length
    ring[i].K = strength_before + dk/2

    qxx_p, qxy_p = ORM_x1(correctors_kick, ring, used_cor_h, used_bpm1)
    qyy_p, qyx_p = ORM_y1(correctors_kick, ring, used_cor_v,  used_bpm2)

    ring[i].K = strength_before - dk/2

    qxx_m, qxy_m = ORM_x1(correctors_kick, ring, used_cor_h, used_bpm1)
    qyy_m, qyx_m = ORM_y1(correctors_kick, ring, used_cor_v,  used_bpm2)

    ring[i].K = strength_before


    qxx = (qxx_p - qxx_m)/  dk/l
    qyy = (qyy_p - qyy_m) / dk/l
    qxy = (qxy_p - qxy_m) / dk/l
    qyx = (qyx_p - qyx_m) / dk/l


    return  qxx, qxy, qyy, qyx


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


    qxx, qxy = ORM_x_G(correctors_kick, ring, used_cor_h, corx_calibration,  used_bpm1,used_bpm2, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)

    qyy, qyx = ORM_y_G(correctors_kick, ring, used_cor_v, cory_calibration,  used_bpm1,used_bpm2, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)


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



    qxx, qxy = ORM_x_G(correctors_kick, ring, used_cor_h, corx_calibration,  used_bpm1,used_bpm2, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)

    qyy, qyx = ORM_y_G(correctors_kick, ring, used_cor_v, cory_calibration,  used_bpm1,used_bpm2, bpmx_calibration, bpmy_calibration, Hor_noise, Ver_noise)


    return  qxx, qxy, qyy, qyx



















def simulateShiftErrors(lattice, shiftx,shifty, elementsInd=None, sigmaCut=None, relative=False):
    for i in elementsInd:
        a =  generateTrucatedGaussian(mean=0, sigma=shiftx, cutValue=sigmaCut*shiftx)
        b =  generateTrucatedGaussian(mean=0, sigma=shifty, cutValue=sigmaCut*shifty)
        at.shift_elem(lattice[i],a, b,
                      relative)

def simulateTilttErrors(lattice, rots, elementsInd=None, sigmaCut=None, relative=True):
    for i in elementsInd:
        a = generateTrucatedGaussian(mean=0, sigma=rots, cutValue=sigmaCut * rots)
        at.tilt_elem(lattice[i], rots=a, relative=relative)


def generateTrucatedGaussian(mean, sigma, cutValue):
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

def getBetaBeat(ring, twiss, elements_indexes, makeplot):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    Beta_x = twiss_error.beta[:, 0]
    Beta_y = twiss_error.beta[:, 1]
    bx = np.array((twiss_error.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0])
    by = np.array((twiss_error.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1])
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if makeplot == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, bx)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal beta beating')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, by)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical beta beating')
        plt.show()

    return bx_rms, by_rms

def corrections_plots(ring ,twiss, twiss_err, plot):
    s_pos_err = twiss_err.s_pos
    Beta_x_err = twiss_err.beta[:, 0]
    Beta_y_err = twiss_err.beta[:, 1]
    closed_orbitx_err = twiss_err.closed_orbit[:, 0]
    closed_orbity_err = twiss_err.closed_orbit[:, 2]
    elements_indexes = get_refpts(ring,at.elements.Monitor)

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

def defineJacobianMatrices(C0x, C0y, C0xy, C0yx, Cx, Cy, Cxy, Cyx, dCx, dCy, dCxy,dCyx):
    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements
    Ax = np.zeros([Nk, Nk])
    Ay = np.zeros([Nk, Nk])
    Axy = np.zeros([Nk, Nk])
    Ayx = np.zeros([Nk, Nk])
    A = np.zeros([4 * Nk, Nk])

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    Bxy = np.zeros([Nk, 1])
    Byx = np.zeros([Nk, 1])
    B = np.zeros([4 * Nk, 1])

    Dx = (Cx[0:Nm, :] - C0x[0:Nm, :])
    Dy = (Cy[0:Nm, :] - C0y[0:Nm, :])
    Dxy = (Cxy[0:Nm, :] - C0xy[0:Nm, :])
    Dyx = (Cyx[0:Nm, :] - C0yx[0:Nm, :])


    tmp = np.sum(dCx, axis=1)          # Sum over i and j for all planes
    Ax = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)          # Sum over i and j for all planes
    Ay = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCxy, axis=1)          # Sum over i and j for all planes
    Axy = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCyx, axis=1)          # Sum over i and j for all planes
    Ayx = tmp @ tmp.T                   # Final sum over k for all planes

    # Fill A with its components
    A = np.zeros([4 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:2*Nk, :] = Ay
    A[2*Nk:3*Nk, :] = Axy
    A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))
        Bxy[i] = np.sum(np.dot(dCxy[i], Dxy.T))
        Byx[i] = np.sum(np.dot(dCyx[i], Dyx.T))
        B[i] = Bx[i]
        B[i + Nk] = By[i]
        B[i + 2 * Nk] = Bxy[i]
        B[i + 3 * Nk] = Byx[i]

    return A, B

def defineJacobianMatrices_w(w, C0x, C0y, C0xy, C0yx, Cx, Cy, Cxy, Cyx, dCx, dCy, dCxy,dCyx):
    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    Bxy = np.zeros([Nk, 1])
    Byx = np.zeros([Nk, 1])
    B = np.zeros([4 * Nk, 1])

    Dx = (Cx[0:Nm, :] - C0x[0:Nm, :])
    Dy = (Cy[0:Nm, :] - C0y[0:Nm, :])
    Dxy = (Cxy[0:Nm, :] - C0xy[0:Nm, :])
    Dyx = (Cyx[0:Nm, :] - C0yx[0:Nm, :])

    tmp = np.sum(dCx, axis=1)

    # Sum over i and j for all planes
    Ax = tmp @ w @ tmp.T                # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)          # Sum over i and j for all planes
    Ay = tmp @ w @ tmp.T                    # Final sum over k for all planes

    tmp = np.sum(dCxy, axis=1)          # Sum over i and j for all planes
    Axy = tmp @ w @ tmp.T                      # Final sum over k for all planes

    tmp = np.sum(dCyx, axis=1)          # Sum over i and j for all planes
    Ayx = tmp @ w @ tmp.T                     # Final sum over k for all planes

    # Fill A with its components
    A = np.zeros([4 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:2*Nk, :] = Ay
    A[2*Nk:3*Nk, :] = Axy
    A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(np.dot(dCx[i], w), Dx.T))
        By[i] = np.sum(np.dot(np.dot(dCy[i], w), Dy.T))
        Bxy[i] = np.sum(np.dot(np.dot(dCxy[i], w), Dxy.T))
        Byx[i] = np.sum(np.dot(np.dot(dCyx[i], w), Dyx.T))
        B[i] = Bx[i]
        B[i + Nk] = By[i]
        B[i + 2 * Nk] = Bxy[i]
        B[i + 3 * Nk] = Byx[i]

    return A, B


def defineJacobianMatrices_w(C0x, C0y, C0xy, C0yx, Cx, Cy, Cxy, Cyx, dCx, dCy, dCxy,dCyx, sigmax, sigmay):
    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    Bxy = np.zeros([Nk, 1])
    Byx = np.zeros([Nk, 1])
    B = np.zeros([4 * Nk, 1])

    Dx = (Cx[0:Nm, :] - C0x[0:Nm, :])
    Dy = (Cy[0:Nm, :] - C0y[0:Nm, :])
    Dxy = (Cxy[0:Nm, :] - C0xy[0:Nm, :])
    Dyx = (Cyx[0:Nm, :] - C0yx[0:Nm, :])

    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements
    Ax = np.zeros([Nk, Nk])
    Ay = np.zeros([Nk, Nk])
    Axy = np.zeros([Nk, Nk])
    Ayx = np.zeros([Nk, Nk])
    A = np.zeros([4 * Nk, Nk])

    for i in range(Nk):
        #print('done A:', 100.* i ,'%')
        for j in range(Nk):
            Ax[i, j] = np.sum(np.dot(np.dot(sigmax, dCx[i]), dCx[j].T))
            Ay[i, j] = np.sum(np.dot(np.dot(sigmay, dCy[i]), dCy[j].T))
            Axy[i, j] = np.sum(np.dot(np.dot(sigmay, dCxy[i]), dCxy[j].T))
            Ayx[i, j] = np.sum(np.dot(np.dot(sigmax, dCyx[i]), dCyx[j].T))
        A[i, :] = Ax[i, :]
        A[i + Nk, :] = Ay[i, :]
        A[i + 2 * Nk, :] = Axy[i, :]
        A[i + 3 * Nk, :] = Ayx[i, :]

    # Fill A with its components
    A = np.zeros([4 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:2*Nk, :] = Ay
    A[2*Nk:3*Nk, :] = Axy
    A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(np.dot(sigmax, dCx[i]), Dx.T))
        By[i] = np.sum(np.dot(np.dot(sigmay, dCy[i]), Dy.T))
        Bxy[i] = np.sum(np.dot(np.dot(sigmay, dCyx[i]), Dyx.T))
        Byx[i] = np.sum(np.dot(np.dot(sigmax, dCxy[i]), Dxy.T))
        B[i] = Bx[i]
        B[i + Nk] = By[i]
        B[i + 2 * Nk] = Bxy[i]
        B[i + 3 * Nk] = Byx[i]

    return A, B


def getInverse(A, B,Nk, sCut, showPlots='False'):
    u, s, v = np.linalg.svd(A, full_matrices=True)

    smat = 0.0 * A
    si = s ** -1
    n_sv = sCut
    si[n_sv:] *= 0.0
    print("number of singular values {}".format(len(si)))
    smat[:Nk, :Nk] = np.diag(si)
    Ai = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
    r = (np.dot(Ai, B)).reshape(-1)
    e = np.dot(A, r).reshape(-1) - B.reshape(-1)

    if showPlots == 'True':
       plt.plot(si, 'd--')
       plt.title('singular value')
       plt.show()

       plot(r, 'd')
       plt.xlabel('s(m)')
       plt.ylabel(r'$\frac{\Delta k}{k}%$')
       plt.title('relative quads value')
       plt.show()

       plt.plot(e)
       plt.title('correction error')
       plt.show()
    return r

def setCorrection(ring, r ,quadInd):
    for i in range(len(quadInd)):
        qInd = quadInd[i]
        ring[qInd].K += -r[i]

def setCorrection_norm(ring, r, quadInd):
    for i in range(len(quadInd)):
        l = ring[i].Length
        qInd = quadInd[i]
        ring[qInd].K += -r[i]/l


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
   if print_output == True:

       vectors = [bpmx_calibration, bpmy_calibration, corx_calibration, cory_calibration, Hor_bpm_noise, Ver_bpm_noise]
       vector_names = ['bpmx_calibration', 'bpmy_calibration', 'corx_calibration', 'cory_calibration', 'Hor_bpm_noise',
                       'Ver_bpm_noise']
       data = {name: vector for name, vector in zip(vector_names, vectors)}
       df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))
       print(df)

   return bpmx_calibration,  bpmy_calibration,  corx_calibration, cory_calibration, Hor_bpm_noise,Ver_bpm_noise, df

def ORM_x_G(dkick, ring, used_correctors_ind, corCal, used_bpm1,used_bpm2, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey):
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
    Cxx = (np.squeeze(cxx_p)-np.squeeze(cxx_m))/dkick
    Cxy = (np.squeeze(cxy_p)-np.squeeze(cxy_m))/dkick

    return (Cxx), (Cxy)


def ORM_y_G(dkick, ring, used_correctors_ind, corCal,  used_bpm1,used_bpm2, bpm_calibrationx, bpm_calibrationy, bpm_noisex, bpm_noisey):
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
    Cyy = (np.squeeze(cyy_p)-np.squeeze(cyy_m))/dkick
    Cyx = (np.squeeze(cyx_p)-np.squeeze(cyx_m))/dkick

    return (Cyy), (Cyx)



def loco_correction(objective_function, initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, Jt, lengths,   including_fit_parameters, method='lm', verbose=2, max_iterations=1000, eps=1e-6,W = 1, show_plot = True):

    if method == 'lm':
        result = least_squares(objective_function, initial_guess0, method=method, verbose=verbose)#, xtol= 1e-2)
        params_to_check = calculate_parameters(result.x, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,including_fit_parameters)
        return result.x
    else:
        if method == 'ng':
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
                    # Co = orbit_response_matrix_model * delta_x
                    Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
                    model += Co

                if 'bpm' in including_fit_parameters:
                    delta_y = initial_guess0[len_quads + len_corr:]
                    J3 = J[len_quads + len_corr:]
                    #G = orbit_response_matrix_model * delta_y[:, np.newaxis]
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

                #if show_plot == True:

                    #e = np.dot(initial_guess0, J) - t2

                    #plt.plot(e)
                    #plt.title('correction error')
                    #plt.show()

        #params_to_check = calculate_parameters(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,including_fit_parameters)


        return initial_guess0 #, params_to_check


def objective(delta_params, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters, W):

    D = orbit_response_matrix_measured - orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    residuals = D
    if 'quads' in including_fit_parameters:

        delta_g = delta_params[:len_quads]
        J1 = J[:len_quads]
        B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
        residuals -= B


    if 'cor' in including_fit_parameters:

        delta_x = delta_params[len_quads:len_quads + len_corr]
        J2= J[len_quads:len_quads + len_corr]
        #Co = orbit_response_matrix_model * delta_x
        Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
        residuals -= Co


    if 'bpm' in including_fit_parameters:


        delta_y = delta_params[len_quads + len_corr:]
        J3= J[len_quads + len_corr:]
        #G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)
        residuals -= G


    residuals = np.dot(residuals, np.sqrt(W))


    return residuals.ravel()



def calculate_parameters(parameters, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters):
    model = orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        delta_g = parameters[:len_quads]
        J1 = J[:len_quads]
        B = np.sum([J1[k] * delta_g[k] for k in range(len(J1))], axis=0)
        model += B

    if 'cor' in including_fit_parameters:
        delta_x = parameters[len_quads:len_quads + len_corr]
        J2 = J[len_quads:len_quads + len_corr]
        # Co = orbit_response_matrix_model * delta_x
        Co = np.sum([J2[k] * delta_x[k] for k in range(len(J2))], axis=0)
        model += Co

    if 'bpm' in including_fit_parameters:
        delta_y = parameters[len_quads + len_corr:]
        J3 = J[len_quads + len_corr:]
        # G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        G = np.sum([J3[k] * delta_y[k] for k in range(len(J3))], axis=0)
        model += G


    residuals = orbit_response_matrix_measured- model
    # Calculate R-squared
    r_squared = r2_score(orbit_response_matrix_measured, model)

    # Calculate RMSE
    rms = sqrt(mean_squared_error(orbit_response_matrix_measured,model))

    params_to_check_ = {

        'r_squared': r_squared,
        'rmse': rms,
    }

    return params_to_check_