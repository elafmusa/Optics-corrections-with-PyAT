from at import *
import at.plot
from pylab import *
from scipy.optimize import least_squares

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
        #cor_index = cor_index[0]

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
    Cyy = (np.squeeze(cyy_p)-np.squeeze(cyy_m))/dkick
    Cyx = (np.squeeze(cyx_p)-np.squeeze(cyx_m))/dkick

    return (Cyy), (Cyx)


def ORMs(dkick, ring, used_correctors):
    Cxx = []
    Cxy = []
    Cyy = []
    Cyx = []

    bpm_indexes = get_refpts(ring, at.elements.Monitor)
    _, _, elemdata = at.get_optics(ring, bpm_indexes)
    closed_orbitx0 = elemdata.closed_orbit[:, 0]
    closed_orbity0 = elemdata.closed_orbit[:, 2]

    for i in used_correctors:
        a_x = ring[i].KickAngle[0]
        a_y = ring[i].KickAngle[1]

        ring[i].KickAngle[0] = dkick + a_x
        _, _, elemdata = at.get_optics(ring, bpm_indexes)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        Cxx.append(closed_orbitx)
        Cxy.append(closed_orbity)

        ring[i].KickAngle[0] = a_x
        ring[i].KickAngle[1] = dkick + a_y
        _, _, elemdata = at.get_optics(ring, bpm_indexes)
        closed_orbitx = elemdata.closed_orbit[:, 0] - closed_orbitx0
        closed_orbity = elemdata.closed_orbit[:, 2] - closed_orbity0
        Cyy.append(closed_orbity)
        Cyx.append(closed_orbitx)

        ring[i].KickAngle[1] = a_y

    Cxx = np.squeeze(Cxx) / dkick
    Cxy = np.squeeze(Cxy) / dkick
    Cyy = np.squeeze(Cyy) / dkick
    Cyx = np.squeeze(Cyx) / dkick

    return Cxx, Cxy, Cyy, Cyx

import multiprocessing

def generatingQuadsResponseP_sext(ring, C0x, C0y, C0xy, C0yx, correctrs_kick, used_cor_indexes, quads_indexes, dk, debug=False):
    import multiprocessing
    def worker(quad_index):
        if debug:
            print('generating response to quad of index', quad_index)
        C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices_sext(ring, correctrs_kick, used_cor_indexes, quad_index, dk)
        return (C1x - C0x) / dk, (C1y - C0y) / dk, (C1xy - C0xy) / dk, (C1yx - C0yx) / dk

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(worker, quads_indexes)
    pool.close()
    pool.join()

    dCx, dCy, dCxy, dCyx = zip(*results)
    return list(dCx), list(dCy), list(dCxy), list(dCyx)

def quadsSensitivityMatrices_sext(ring, correctors_kick, used_cor_indexes, quad_index, dk):

    strength_before = ring[quad_index].PolynomA[1]
    ring[quad_index].PolynomA[1] +=  dk
    qxx, qxy , qyy, qyx= ORMs(correctors_kick, ring, used_cor_indexes)
    ring[quad_index].PolynomA[1] = strength_before
    return  qxx, qxy, qyy, qyx

def  defineMatrices_eta(C0x, C0y, Cxx_err, Cyy_err, dCx, dCy):
    Nk = len(dCx)  # number of free parameters
    Nm = len(dCx)   # number of measurements
    print('NK:', Nk)
    print('Nm:', Nm)

    Ax = np.zeros([Nk, Nk])
    Ay = np.zeros([Nk, Nk])

    A = np.zeros([2 * Nk, Nk])

    ##

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])

    B = np.zeros([2 * Nk, 1])

    ##

    Dx = (Cxx_err - C0x)  ### dk ?
    Dy = (Cyy_err - C0y)


    ##

    for i in range(Nk):  ## i represents each quad
        # print('done A:', 100.* i ,'%')
        for j in range(Nk):
            Ax[i, j] = np.sum(np.dot(dCx[i], dCx[j].T))
            Ay[i, j] = np.sum(np.dot(dCy[i], dCy[j].T))

        A[i, :] = Ax[i, :]
        A[i + Nk, :] = Ay[i, :]


    ##

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))


        B[i] = Bx[i]
        B[i + Nk] = By[i]


    return A, B


def generatingQuadsResponse(ring, C0x, C0y, C0xy, C0yx, correctrs_kick,used_cor_x, used_cor_y, quads_indexes, dk, bpm_ind, debug=False):
    dCx = []
    dCy = []
    dCxy = []
    dCyx = []
    for quad_index in quads_indexes:
        if debug == True:
            print('generating response to quad of index', quad_index)
        if ring[quad_index].FamName.startswith('s') or ring[quad_index].FamName.startswith('S'):
            C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices_skew(ring,correctrs_kick,used_cor_x, used_cor_y,  quad_index,bpm_ind, dk)
        else:

           C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices(ring,correctrs_kick,used_cor_x, used_cor_y,  quad_index, bpm_ind, dk)

        dCx.append((C1x - C0x) / dk)
        dCy.append((C1y - C0y) / dk)
        dCxy.append((C1xy - C0xy) / dk)
        dCyx.append((C1yx - C0yx) / dk)
    return dCx, dCy, dCxy, dCyx

def quadsSensitivityMatrices(ring,correctrs_kick,used_cor_x, used_cor_y,  quad_index, bpm_ind, dk):
    strength_before = ring[quad_index].PolynomB[1]
    ring[quad_index].PolynomB[1] = strength_before + dk

    qxx, qxy = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy, qyx = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[quad_index].PolynomB[1] = strength_before
    return  qxx, qxy, qyy, qyx






def generatingQuadsResponse_skew(ring, C0x, C0y, C0xy, C0yx, correctrs_kick,used_cor_x, used_cor_y, quads_indexes, dk, bpm_ind,debug=False):
    dCx = []
    dCy = []
    dCxy = []
    dCyx = []
    for quad_index in quads_indexes:
        if debug == True :
           print('generating response to quad of index', quad_index)
        C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices_skew(ring,correctrs_kick,used_cor_x, used_cor_y,  quad_index, dk,bpm_ind)
        dCx.append((C1x - C0x) / dk)
        dCy.append((C1y - C0y) / dk)
        dCxy.append((C1xy - C0xy) / dk)
        dCyx.append((C1yx - C0yx) / dk)
    return dCx, dCy, dCxy, dCyx


def quadsSensitivityMatrices_skew(ring, correctrs_kick, used_cor_x, used_cor_y, quad_index, dk,bpm_ind):
    strength_before = ring[quad_index].PolynomA[1]
    strength_before = ring[quad_index].PolynomA[1]
    ring[quad_index].PolynomA[1] = strength_before + dk
    # qxx, qxy, qyy, qyx = ORMs(correctrs_kick, ring,used_cor_indexes)

    qxx, qxy = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy, qyx = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[quad_index].PolynomA[1] = strength_before
    return qxx, qxy, qyy, qyx


def quadsSensitivityMatrices_skew_no(ring, correctrs_kick, used_cor_x, used_cor_y, quad_index, bpm_ind, dk):

    i = quad_index

    strength_before = ring[i].PolynomA[1]
    l = ring[i].Length
    ring[i].PolynomA[1] = strength_before + dk/2

    qxx_p, qxy_p = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy_p, qyx_p = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[i].PolynomA[1] = strength_before - dk/2

    qxx_m, qxy_m = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy_m, qyx_m = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[i].PolynomA[1] = strength_before


    qxx = (qxx_p - qxx_m)/  dk/l
    qyy = (qyy_p - qyy_m) / dk/l
    qxy = (qxy_p - qxy_m) / dk/l
    qyx = (qyx_p - qyx_m) / dk/l


    return  qxx, qxy, qyy, qyx


def quadsSensitivityMatrices_n(ring,correctrs_kick,used_cor_x, used_cor_y,  quad_index, bpm_ind, dk):
    i = quad_index

    strength_before = ring[i].K
    l = ring[i].Length
    ring[i].K = strength_before + dk / 2

    qxx_p, qxy_p = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy_p, qyx_p = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[i].K = strength_before - dk / 2

    qxx_m, qxy_m = ORM_x1(correctrs_kick, ring, used_cor_x, bpm_ind)
    qyy_m, qyx_m = ORM_y1(correctrs_kick, ring, used_cor_y, bpm_ind)

    ring[i].K = strength_before

    qxx = (qxx_p - qxx_m) / dk / l
    qyy = (qyy_p - qyy_m) / dk / l
    qxy = (qxy_p - qxy_m) / dk / l
    qyx = (qyx_p - qyx_m) / dk / l

    return  qxx, qxy, qyy, qyx



def generatingQuadsResponse_eta(ring, C0x, C0y, quads_indexes, dk):
    dCx = []
    dCy = []


    for quad_index in quads_indexes:
        if ring[quad_index].FamName.startswith('s') or ring[quad_index].FamName.startswith('S'):
            C1x, C1y = quadsSensitivityMatrices_skew_eta(ring,quad_index, dk)
        else:

           C1x, C1y = quadsSensitivityMatrices_eta(ring,quad_index, dk)

        dCx.append((C1x - C0x) / dk)
        dCy.append((C1y - C0y) / dk)
        #dCx.append(C1x)
        #dCy.append(C1y)

    return dCx, dCy

def quadsSensitivityMatrices_skew_eta(ring,quad_index, dk):

    strength_before = ring[quad_index].PolynomA[1]
    ring[quad_index].PolynomA[1] = strength_before + dk
    bpm_indices = get_refpts(ring, at.elements.Monitor)
    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=bpm_indices)


    Eta_xx = lindata['dispersion'][:, 0]
    Eta_yy = lindata['dispersion'][:, 2]


    ring[quad_index].PolynomA[1] = strength_before
    return  Eta_xx, Eta_yy#, qyy, qyx


def quadsSensitivityMatrices_eta(ring,quad_index, dk):

    strength_before = ring[quad_index].PolynomB[1]
    ring[quad_index].PolynomB[1] = strength_before + dk

    bpm_indices = get_refpts(ring, at.elements.Monitor)
    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=bpm_indices)

    Eta_xx = lindata['dispersion'][:, 0]
    Eta_yy = lindata['dispersion'][:, 2]


    ring[quad_index].PolynomB[1] = strength_before
    return  Eta_xx, Eta_yy#, qyy, qyx














def generateTrucatedGaussian(mean, sigma, cutValue):
    numberFound = False
    while(numberFound == False):
        a = (mean + sigma * np.random.randn())
        if - cutValue > a or a > cutValue:
            numberFound = False
        else:
            numberFound = True
            return a

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


def generatingQuadsResponse_eta(ring, C0x, C0y, quads_indexes, dk):
    dCx = []
    dCy = []


    for quad_index in quads_indexes:
        if ring[quad_index].FamName.startswith('s') or ring[quad_index].FamName.startswith('S'):
            C1x, C1y = quadsSensitivityMatrices_skew_eta(ring,quad_index, dk)
        else:

           C1x, C1y = quadsSensitivityMatrices_eta(ring,quad_index, dk)

        dCx.append((C1x - C0x) / dk)
        dCy.append((C1y - C0y) / dk)
        #dCx.append(C1x)
        #dCy.append(C1y)

    return dCx, dCy

def quadsSensitivityMatrices_skew_eta(ring,quad_index, dk):

    strength_before = ring[quad_index].PolynomA[1]
    ring[quad_index].PolynomA[1] = strength_before + dk
    bpm_indices = get_refpts(ring, at.elements.Monitor)

    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=bpm_indices)


    Eta_xx = lindata['dispersion'][:, 0]
    Eta_yy = lindata['dispersion'][:, 2]


    ring[quad_index].PolynomA[1] = strength_before
    return  Eta_xx, Eta_yy#, qyy, qyx


def quadsSensitivityMatrices_eta(ring,quad_index, dk):

    strength_before = ring[quad_index].PolynomB[1]
    ring[quad_index].PolynomB[1] = strength_before + dk
    bpm_indices = get_refpts(ring, at.elements.Monitor)

    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=bpm_indices)

    Eta_xx = lindata['dispersion'][:, 0]
    Eta_yy = lindata['dispersion'][:, 2]


    ring[quad_index].PolynomB[1] = strength_before
    return  Eta_xx, Eta_yy#, qyy, qyx


def generateTrucatedGaussian(mean, sigma, cutValue):
    numberFound = False
    while(numberFound == False):
        a = (mean + sigma * np.random.randn())
        if - cutValue > a or a > cutValue:
            numberFound = False
        else:
            numberFound = True
            return a

def simulateFieldErrors(lattice, gradErr, elementsInd):
    for i in elementsInd:
        a = (1 + gradErr * np.random.randn())
        lattice[i].K *= a

def simulateFieldErrors_s(lattice, gradErr, elementsInd):
    for i in elementsInd:
        a = (1 + gradErr * np.random.randn())
        lattice[i].H *= a

def simulateFieldErrors_b(lattice, gradErr, elementsInd):
    for i in elementsInd:
        a = (1 + gradErr * np.random.randn())
        lattice[i].H *= a



def rms_orbits(ring, elements_indexes, makeplot):
    _, _, elemdata = at.get_optics(ring, elements_indexes)
    closed_orbitx = elemdata.closed_orbit[:, 0]
    closed_orbity = elemdata.closed_orbit[:, 2]
    s_pos = elemdata.s_pos

    if makeplot == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, closed_orbitx / 1.e-06)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"closed_orbit x [$\mu$m]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title("Closed orbit x")
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, closed_orbity / 1.e-06)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"closed_orbit y [$\mu$m]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title("Closed orbit y")
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
        ax.plot(s_pos, Beta_x)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal beta')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_y)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical beta')
        plt.show()

    return bx_rms, by_rms


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
    if plot == 'eta':
        bx_err = np.array((twiss_err.dispersion[:, 0] - twiss.dispersion[:, 0]) / twiss.dispersion[:, 0])
        by_err = np.array((twiss_err.dispersion[:, 2] - twiss.dispersion[:, 2]) / twiss.dispersion[:, 2])
        bx = np.array((twiss_cor.dispersion[:, 0] - twiss.dispersion[:, 0]) / twiss.dispersion[:, 0])
        by = np.array((twiss_cor.dispersion[:, 2] - twiss.dispersion[:, 2]) / twiss.dispersion[:, 2])

        fig = plt.figure()
        plt.plot(twiss.s_pos, bx_err, label='before correction')
        plt.plot(twiss.s_pos, bx, label='after correction')
        plt.xlabel('s[m]')
        plt.ylabel(r'$\frac{\Delta \eta_x}{\eta_x}%$')
        plt.title("Horizontal dispersion errors")
        plt.legend(loc='upper left')
        fig = plt.figure()
        plt.plot(twiss.s_pos, by_err, label='before correction')
        plt.plot(twiss.s_pos, by, label='after correction')
        plt.xlabel('s[m]')
        plt.ylabel(r'$\frac{\Delta \eta_y}{\eta_y}%$')
        plt.title("Vertical dispersion errors")
        plt.legend(loc='upper left')
        plt.show()

def defineJacobianMatrices(Cx0, Cy0, Cxy0, Cyx0, Cx, Cy, Cxy, Cyx, dCx, dCy, dCxy,dCyx):
    Nk = len(dCx)  # number of free parameters
    Nm = len(Cx0)  # number of measurements


    X = np.zeros((Cx0.shape[0] + Cy0.shape[0], Cx0.shape[1] + Cy0.shape[1]))
    X[:Cx0.shape[0], :Cx0.shape[1]] = Cx0
    X[:Cx0.shape[0], Cx0.shape[1]:] = Cxy0
    X[Cx0.shape[0]:, :Cx0.shape[1]] = Cyx0
    X[Cx0.shape[0]:, Cx0.shape[1]:] = Cy0

    Y = np.zeros((Cx.shape[0] + Cy.shape[0], Cx.shape[1] + Cy.shape[1]))
    Y[:Cx.shape[0], :Cx.shape[1]] = Cx
    Y[:Cx.shape[0], Cx.shape[1]:] = Cxy
    Y[Cx.shape[0]:, :Cx.shape[1]] = Cyx
    Y[Cx.shape[0]:, Cx.shape[1]:] = Cy

    D = Y - X

    J = np.zeros((len(dCx), Cx.shape[0] + Cy.shape[0], Cx.shape[1] + Cy.shape[1]))
    J[:, :Cx.shape[0], :Cx.shape[1]] = dCx
    J[:, :Cx.shape[0], Cx.shape[1]:] = dCxy
    J[:, Cx.shape[0]:, :Cx.shape[1]] = dCyx
    J[:, Cx.shape[0]:, Cx.shape[1]:] = dCy

    B = np.zeros([Nk, 1])
    for i in range(Nk):
        B[i] = np.sum(np.dot(J[i], D.T))

    tmp = np.sum(J, axis=1)
    A = tmp @ tmp.T


    return A, B


def  defineMatrices_eta(C0x, C0y, Cxx_err, Cyy_err, dCx, dCy):
    Nk = len(dCx)  # number of free parameters
    Nm = len(dCx)   # number of measurements
    print('NK:', Nk)
    print('Nm:', Nm)

    Ax = np.zeros([Nk, Nk])
    Ay = np.zeros([Nk, Nk])

    A = np.zeros([2 * Nk, Nk])

    ##

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])

    B = np.zeros([2 * Nk, 1])

    ##

    Dx = (Cxx_err - C0x)  ### dk ?
    Dy = (Cyy_err - C0y)


    ##

    for i in range(Nk):  ## i represents each quad
        # print('done A:', 100.* i ,'%')
        for j in range(Nk):
            Ax[i, j] = np.sum(np.dot(dCx[i], dCx[j].T))
            Ay[i, j] = np.sum(np.dot(dCy[i], dCy[j].T))

        A[i, :] = Ax[i, :]
        A[i + Nk, :] = Ay[i, :]


    ##

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))


        B[i] = Bx[i]
        B[i + Nk] = By[i]


    return A, B


def defineJacobianMatrices_(C0x, C0y, Cx, Cy, dCx, dCy):
    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    B = np.zeros([2 * Nk, 1])

    Dx = (Cx[0:Nm, :] - C0x[0:Nm, :])
    Dy = (Cy[0:Nm, :] - C0y[0:Nm, :])

    tmp = np.sum(dCx, axis=1)  # Sum over i and j for all planes
    Ax = tmp @ tmp.T  # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)  # Sum over i and j for all planes
    Ay = tmp @ tmp.T  # Final sum over k for all planes

    # Fill A with its components
    A = np.zeros([2 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:, :] = Ay

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))

        B[i] = Bx[i]
        B[i + Nk] = By[i]


    return A, B



def defineJacobianMatrices1(C0x, C0y, C0xy, C0yx, Cx, Cy, Cxy, Cyx, dCx, dCy, dCxy,dCyx):
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











def defineJacobianMatrices_noCoupling(C0x, C0y, Cxx_err, Cyy_err, dCx, dCy):

    Nk = len(dCx)  # number of free parameters
    Nm = len(C0x)  # number of measurements


    Dx = (Cxx_err[0:Nm, :] - C0x[0:Nm, :])  ### dk ?
    Dy = (Cyy_err[0:Nm, :] - C0y[0:Nm, :])

    ##

    tmp = np.sum(dCx, axis=1)  # Sum over i and j for all planes
    Ax = tmp  @ tmp.T  # Final sum over k for all planes
    tmp = np.sum(dCy, axis=1)  # Sum over i and j for all planes
    Ay = tmp @ tmp.T  # Final sum over k for all planes

    A = np.zeros([2 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:, :] = Ay


    ##


    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])

    B = np.zeros([2 * Nk, 1])

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))

    B[:Nk, :] = Bx
    B[Nk:, :] = By


    return A, B

def getInverse(A, B,Nk, sCut, showPlots=False):
    u, s, v = np.linalg.svd(A, full_matrices=True)
    smat = 0.0 * A
    si = s ** -1
    n_sv = sCut
    si[n_sv:] *= 0.0
    #print("number of singular values {}".format(len(si)))
    smat[:Nk, :Nk] = np.diag(si)
    #print('A' + str(A.shape), 'B' + str(B.shape), 'U' + str(u.shape), 'smat' + str(smat.shape), 'v' + str(v.shape))
    Ai = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
    r = (np.dot(Ai, B)).reshape(-1)
    e = np.dot(A, r).reshape(-1) - B.reshape(-1)

    if showPlots == True:
        plt.plot(np.log(s), 'd--')
        plt.title('singular value')
        plt.show()

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
        #plt.plot(B)
        #plt.show()


    return r

def setCorrection(ring, r ,quadInd):
    for i in range(len(quadInd)):

        qInd = quadInd[i]

        if ring[qInd].FamName.startswith('s') or ring[qInd].FamName.startswith('S')  :
            ring[qInd].PolynomA[1] += -r[i]
        else:
            ring[qInd].K += -r[i]

def setCorrection_skew(ring, r ,quadInd):
    for i in range(len(quadInd)):
        qInd = quadInd[i]
        ring[qInd].PolynomA[1]+= -r[i]


def dynamicAperture(ring, eh, ev, dr, dp, nturns, xp0, yp0,showPlot=False, debug=False):
    elements_indexes = get_refpts(ring, "*")
    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=elements_indexes[0])
    beta_x = lindata['beta'][:, 0]
    beta_y = lindata['beta'][:, 1]
    sigmax = sqrt(beta_x[0] * eh)
    sigmay = sqrt(beta_y[0] * ev)

    x_a = []
    y_a = []
    theta = 0
    tol = 1.e-12
    t0 = time.time()
    while (theta < 180):
        if debug == True:
           print(f"theta={theta}")
        r0 = 0
        r = find_line_DA(ring, theta, r0, dr, dp, nturns, xp0, yp0, sigmax,
                               sigmay)
        max_value_x = 0.0
        max_value_y = 0.0
        b = r - dr
        if r > r0 + tol:
            r_1 = find_line_DA(ring, theta, b, dr, dp, nturns, xp0, yp0,
                                     sigmax, sigmay)
        else:
            r_1 = r0

        x0 = r_1 * cos(np.radians(theta))
        y0 = r_1 * sin(np.radians(theta))
        #if debug == True:
        #   print(f"found DA {x0} {y0}")
        x_a.append(x0)
        y_a.append(y0)
        #if debug == True:
        #   print(f"DA radius is {r_1}")
        theta += 10



    if showPlot == True:
        plt.scatter(x_a, y_a)
        a, = plt.plot(x_a, y_a, label=r'$\frac{\Delta p}{p}=0.0$')
        plt.legend(handles=[a], loc='lower right')
        plt.xlabel('x(m)/sigma_x')
        plt.ylabel('y(m)/sigma_y')
        plt.title('FCCee_tt DA')

        plt.show()

    #t1 = time.time()
    #print(f"Execution time: {t1 - t0} sec")
    return x_a, y_a


def find_line_DA(ring, theta, r0, dr, dp, nturns, xp0, yp0, sigmax, sigmay):
    max_value_x = 0.0
    max_value_y = 0.0
    radius = r0
    while (not np.isnan(max_value_x) and not np.isnan(max_value_y)):

        x0 = radius * cos(np.radians(theta)) * sigmax
        y0 = radius * sin(np.radians(theta)) * sigmay
        pin1 = [np.array([x0]), np.array([xp0])]
        pin2 = [np.array([y0]), np.array([yp0])]
        pin3 = [np.array([dp])]
        new_pin = np.concatenate((pin1, pin2, pin3, [np.zeros(1)]))
        pout = at.tracking.deprecated.lattice_pass(ring, new_pin.copy(), nturns, refpts=[len(ring)])
        #pout = lattice_track(ring, new_pin.copy(), nturns, refpts=[len(ring)])
        x = pout[0, 0, 0, :]
        xp = pout[1, 0, 0, :]
        y = pout[2, 0, 0, :]
        yp = pout[3, 0, 0, :]
        max_value_x = np.max(x)
        max_value_y = np.max(y)

        if not np.isnan(max_value_x) and not np.isnan(max_value_y): radius += dr

    return radius
def loco_correction(objective_function, initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, Jt, lengths, including_fit_parameters, method='lm', verbose=2, max_iterations=1000, eps=1e-6):
    from scipy.optimize import least_squares
    from sklearn.metrics import r2_score, mean_squared_error
    if method == 'lm':
        result = least_squares(objective_function, initial_guess0, method=method, verbose=verbose)
        params_to_check = calculate_parameters(result.x, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,including_fit_parameters)
        return result, params_to_check
    else:
        if method == 'ng':
            Iter = 0
            while True:
                Iter += 1

                if max_iterations is not None and Iter > max_iterations:
                    break

                model = orbit_response_matrix_model
                if 'quads' in including_fit_parameters:
                    len_quads = lengths[0]
                    delta_g = initial_guess0[:len_quads]
                    B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
                    model += B

                if 'cor' in including_fit_parameters:
                    len_corr = lengths[1]
                    delta_x = initial_guess0[len_quads:len_quads + len_corr]
                    Co = orbit_response_matrix_model * delta_x
                    model += Co

                if 'bpm' in including_fit_parameters:
                    len_bpm = lengths[2]
                    delta_y = initial_guess0[len_quads + len_corr:]
                    G = orbit_response_matrix_model * delta_y[:, np.newaxis]
                    model += G


                r = orbit_response_matrix_measured - model

                t2 = np.zeros([len(initial_guess0), 1])
                for i in range(len(initial_guess0)):
                    t2[i] = np.sum(np.dot(J[i], r.T))

                t3 = (np.dot(Jt, t2)).reshape(-1)
                initial_guess1 = initial_guess0 + t3 #+
                t4 = abs(initial_guess1 - initial_guess0)

                if max(t4) <= eps:
                    break
                initial_guess0 = initial_guess1

            # params_to_check_ = calculate_parameters(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,
            #                     including_fit_parameters,W)

            delta_g = initial_guess0[:len_quads]
            #delta_x = initial_guess0[len_quads:len_quads + len_corr]
            #delta_y = initial_guess0[len_quads + len_corr:]

            D = orbit_response_matrix_measured - orbit_response_matrix_model
            B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
            #Co = orbit_response_matrix_model * delta_x
            #G = orbit_response_matrix_model * delta_y[:, np.newaxis]
            model = orbit_response_matrix_model + B
            residuals = orbit_response_matrix_measured - model

            r_squared = r2_score(orbit_response_matrix_measured, model)
            rms = sqrt(mean_squared_error(orbit_response_matrix_measured, model))

            params_to_check_ = {
                #'residulas': residuals,
           'r_squared': r_squared,
            'rmse': rms,
            }


        return initial_guess0, params_to_check_


def objective(delta_params, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters):

    D = orbit_response_matrix_measured - orbit_response_matrix_model
    residuals = D
    if 'quads' in including_fit_parameters:
        len_quads = lengths[0]
        delta_g = delta_params[:len_quads]
        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        residuals -= B

    if 'cor' in including_fit_parameters:
        len_corr = lengths[1]
        delta_x = delta_params[len_quads:len_quads + len_corr]
        Co = orbit_response_matrix_model * delta_x
        residuals -= Co

    if 'bpm' in including_fit_parameters:
        len_bpm = lengths[2]
        delta_y = delta_params[len_quads + len_corr:]
        G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        residuals -= G

    residuals = residuals

    return residuals.ravel()



def calculate_parameters(parameters, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters):
    model = orbit_response_matrix_model
    len_quads = lengths[0]
    #len_corr = lengths[1]
    #len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        delta_g = parameters[:len_quads]
        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        model += B



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