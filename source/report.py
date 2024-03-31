from .building_blocks import *
from .project_code import *
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import time
from concurrent.futures import ProcessPoolExecutor

def createTestFlatVol(S, r, q, smileInterpMethod):
    pillars = [0.02, 0.04, 0.06, 0.08, 0.16, 0.25, 0.75, 1.0, 1.5, 2, 3, 5]
    atmvols = [0.155, 0.1395, 0.1304, 0.1280, 0.1230, 0.1230, 0.1265, 0.1290, 0.1313, 0.1318, 0.1313, 0.1305, 0.1295]
    bf25s = np.zeros(len(atmvols))
    rr25s = np.zeros(len(atmvols))
    bf10s = np.zeros(len(atmvols))
    rr10s = np.zeros(len(atmvols))
    smiles = [smileFromMarks(pillars[i], S, r, q, atmvols[i], bf25s[i], rr25s[i], bf10s[i], rr10s[i], smileInterpMethod) for i in range(len(pillars))]
    return ImpliedVol(pillars, smiles)

def createTestImpliedVol(S, r, q, sc, smileInterpMethod):
    pillars = [0.02, 0.04, 0.06, 0.08, 0.16, 0.25, 0.75, 1.0, 1.5, 2, 3, 5]
    atmvols = [0.155, 0.1395, 0.1304, 0.1280, 0.1230, 0.1230, 0.1265, 0.1290, 0.1313, 0.1318, 0.1313, 0.1305, 0.1295]
    bf25s = [0.0016, 0.0016, 0.0021, 0.0028, 0.0034, 0.0043, 0.0055, 0.0058, 0.0060, 0.0055, 0.0054, 0.0050, 0.0045, 0.0043]
    rr25s = [-0.0065, -0.0110, -0.0143, -0.0180, -0.0238, -0.0288, -0.0331, -0.0344, -0.0349, -0.0340, -0.0335, -0.0330, -0.0330]
    bf10s = [0.0050, 0.0050, 0.0067, 0.0088, 0.0111, 0.0144, 0.0190, 0.0201, 0.0204, 0.0190, 0.0186, 0.0172, 0.0155, 0.0148]
    rr10s = [-0.0111, -0.0187, -0.0248, -0.0315, -0.0439, -0.0518, -0.0627, -0.0652, -0.0662, -0.0646, -0.0636, -0.0627, -0.0627]
    smiles = [smileFromMarks(pillars[i], S, r, q, atmvols[i], bf25s[i]*sc, rr25s[i]*sc, bf10s[i]*sc, rr10s[i]*sc, smileInterpMethod) for i in range(len(pillars))]
    return ImpliedVol(pillars, smiles)

def plotTestImpliedVolSurface(S, r, q, iv):
    tStart, tEnd = 0.02, 5
    ts = np.arange(tStart, tEnd, 0.1)
    fwdEnd = S*math.exp((r-q)*tEnd)
    kmin = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.1, PayoffType.Put)
    kmax = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.1, PayoffType.Call)
    #print(f'kmin: {kmin}, kmax= {kmax}')
    ks = np.arange(kmin, kmax, 0.01)
    vs = np.ndarray((len(ts), len(ks)))
    lv = LocalVol(iv, S, r, q)
    lvs = np.ndarray((len(ts), len(ks)))
    for i in range(len(ts)):
        for j in range(len(ks)):
            vs[i, j] = iv.Vol(ts[i], ks[j])
            lvs[i, j] = lv.LV(ts[i], ks[j])
    hf = plt.figure(figsize=(16, 10), dpi=80)
    ha = hf.add_subplot(121, projection='3d')
    hb = hf.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(ks, ts)
    ha.plot_surface(X, Y, vs)
    ha.set_title("implied vol")
    ha.set_xlabel("strike")
    ha.set_ylabel("T")
    hb.plot_surface(X, Y, lvs)
    hb.set_title("local vol")
    hb.set_xlabel("strike")
    hb.set_ylabel("T")
    plt.show()

## Adding function to plot a single smile
def plotTestImpliedVolSmile(S, r, q, iv, t):
    pos = bisect.bisect_left(iv.ts, t)
    tStart, tEnd = 0.02, 5
    # ts = np.arange(tStart, tEnd, 0.1)
    fwdEnd = S*math.exp((r-q)*tEnd)
    kmin = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.1, PayoffType.Put)
    kmax = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.1, PayoffType.Call)
    #print(f'kmin: {kmin}, kmax= {kmax}')
    ks = np.arange(kmin, kmax, 0.01)
    vs = np.zeros(len(ks))
    lv = LocalVol(iv, S, r, q)
    lvs = np.zeros(len(ks))
    for j in range(len(ks)):
        vs[j] = iv.Vol(t, ks[j])
        lvs[j] = lv.LV(t, ks[j])
    hf = plt.figure(figsize=(18, 8), dpi=80)
    ha = hf.add_subplot(121)
    hb = hf.add_subplot(122)
    ha.plot(ks,vs)
    ha.scatter(iv.smiles[pos].strikemarks,iv.smiles[pos].volmarks, label='Marks', c='red')
    ha.legend()
    ha.set_title(f'Implied vol for t= {t}')
    ha.set_xlabel("strike")
    hb.plot(ks,lvs)
    hb.set_title(f'Local vol for t= {t}')
    hb.set_xlabel("strike")
    plt.show()

# Adding a function to plot Call Price and PDF Surface

def plotTestAFCallPrices_PDF(iv):
    ts = iv.ts
    tStart, tEnd = ts[0],ts[-1]
    smiles = iv.smiles
    ks=smiles[0].ks
    cs = np.ndarray((len(ts), len(ks)))
    ps = np.ndarray((len(ts), len(ks)))
    for i in range(len(ts)):
        for j in range(len(ks)):
            cs[i, j] = smiles[i].cs[j]
            ps[i, j] = smiles[i].ps[j]
    hf = plt.figure(figsize=(16, 10), dpi=80)
    ha = hf.add_subplot(121, projection='3d')
    hb = hf.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(ks, ts)
    ha.plot_surface(X, Y, cs)
    ha.set_title("Call Prices Surface")
    ha.set_xlabel("Strike")
    ha.set_ylabel("T")
    hb.plot_surface(X, Y, ps)
    hb.set_title("PDF Surface")
    hb.set_xlabel("Strike")
    hb.set_ylabel("T")
    plt.show()

## Adding function to plot call prices and PDF for single pillar
def plotTestAFCallPrices_PDF_Single(iv, t):
    i= bisect.bisect_left(iv.ts, t)
    smiles = iv.smiles
    ks= smiles[0].ks
    cs = np.zeros(len(ks))
    ps = np.zeros(len(ks))
    for j in range(len(ks)):
        cs[j] = smiles[i].cs[j]
        ps[j] = smiles[i].ps[j]
    hf = plt.figure(figsize=(18, 6), dpi=80)
    ha = hf.add_subplot(121)
    hb = hf.add_subplot(122)
    ha.plot(ks,cs)
    ha.set_title(f'Call Prices for t= {t}')
    ha.set_xlabel("Strike")
    ha.set_ylabel("Call Price")
    hb.bar(ks,ps, width=0.0025)
    hb.set_title(f'PDF for t= {t}')
    hb.set_xlabel("Strike")
    hb.set_ylabel("PDF")
    plt.show()
    print(f'Sum of PDF = {sum(ps)*smiles[i].u}')

# Adding a function to plot pdf Surface

# def plotTestAFPDF(iv):
#     ts = iv.ts
#     tStart, tEnd = ts[0],ts[-1]
#     smiles = iv.smiles
#     ks=smiles[0].ks
#     ps = np.ndarray((len(ts), len(ks)))
#     for i in range(len(ts)):
#         for j in range(len(ks)):
#             ps[i, j] = smiles[i].ps[j]
#     hf = plt.figure(figsize=(18, 8), dpi=80)
#     ha = hf.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(ks, ts)
#     ha.plot_surface(X, Y, ps)
#     ha.set_title("PDF")
#     ha.set_xlabel("Strike")
#     ha.set_ylabel("T")
#     plt.show()

# ## Adding function to plot pdf for single pillar
# def plotTestAFPDF_Single(iv, t):
#     i= bisect.bisect_left(iv.ts, t)
#     smiles = iv.smiles
#     ks= smiles[0].ks
#     ps = np.zeros(len(ks))
#     for j in range(len(ks)):
#         ps[j] = smiles[i].ps[j]
#     plt.bar(ks,ps, width=0.0025)
#     plt.xlabel('Strike')
#     plt.ylabel('PDF')
#     # hf = plt.figure(figsize=(8, 6), dpi=80)
#     # ha = hf.add_subplot(111)
#     # ha.bar(ks,ps)
#     # ha.set_title("PDF")
#     # ha.set_xlabel("Strike")
#     plt.show()

# the PDE calibration error report takes a implied volatility surface,
# verifies the pricing error of the pde pricer with local volatility surface
def pdeCalibReport(S0, r, q, impliedVol):
    report_start = time_block_start()
    ts = [0.02, 0.04, 0.06, 1/12.0, 1/6.0, 1/4.0, 1/2.0, 1, 2, 5]
    ds = np.arange(0.1, 1.0, 0.1)
    # ds = np.arange(0.5, 1.7, 0.1)
    err = np.zeros((len(ds), len(ts)))
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(ts)))
    ax.set_xlabel("T")
    ax.set_yticks(np.arange(len(ds)))
    ax.set_ylabel("Put Delta")
    ax.set_xticklabels(map(lambda t : round(t, 2), ts))
    ax.set_yticklabels(map(lambda d : round(d, 1), ds))

    # create local vol surface
    lv = LocalVol(impliedVol, S0, r, q)
    # Loop over data dimensions and create text annotations.
    for i in range(len(ds)):
        for j in range(len(ts)):
            T = ts[j]
            K = strikeFromDelta(S0, r, 0, T, impliedVol.Vol(T, S0*math.exp(r*T)), ds[i], PayoffType.Put)
            payoff = PayoffType.Put
            trade = EuropeanOption("ASSET1", T, K, payoff)
            vol = impliedVol.Vol(ts[j], K)
            bs = bsPrice(S0, r, q, vol, T, K, payoff)
            pde = pdePricerX(S0, r, q, lv, max(50, int(50 * T)), max(50, int(50 * T)), 0.5, trade)
            # normalize error in 1 basis point per 1 unit of stock
            err[i, j] = math.fabs(bs - pde)/S0 * 10000
            ax.text(j, i, round(err[i, j], 1), ha="center", va="center", color="w")
    im = ax.imshow(err)
    ax.set_title("Dupire Calibration PV Error Matrix")
    fig.tight_layout()
    plt.show()
    time_block_end(report_start, "Total Calibration Report")


def pdeCalibReport_AFvsCS(S0, r, q, impliedVolAF, impliedVolCS):
    ts = [0.02, 0.04, 0.06, 1/12.0, 1/6.0, 1/4.0, 1/2.0, 1, 2, 5]
    ds = np.arange(0.1, 1.0, 0.1)
    # ds = np.arange(0.5, 1.7, 0.1)
    err = np.zeros((len(ds), len(ts)))
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(ts)))
    ax.set_xlabel("T")
    ax.set_yticks(np.arange(len(ds)))
    ax.set_ylabel("Put Delta")
    ax.set_xticklabels(map(lambda t : round(t, 2), ts))
    ax.set_yticklabels(map(lambda d : round(d, 1), ds))

    # create local vol surface
    lv_AF = LocalVol(impliedVolAF, S0, r, q)
    lv_CS = LocalVol(impliedVolCS, S0, r, q)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ds)):
        for j in range(len(ts)):
            T = ts[j]
            K = strikeFromDelta(S0, r, 0, T, impliedVolCS.Vol(T, S0*math.exp(r*T)), ds[i], PayoffType.Put)
            payoff = PayoffType.Put
            trade = EuropeanOption("ASSET1", T, K, payoff)
            vol = impliedVolCS.Vol(ts[j], K)
            bs = bsPrice(S0, r, q, vol, T, K, payoff)
            pde = pdePricerX(S0, r, q, lv_AF, max(50, int(50 * T)), max(50, int(50 * T)), 0.5, trade)
            # normalize error in 1 basis point per 1 unit of stock
            err[i, j] = math.fabs(bs - pde)/S0 * 10000
            ax.text(j, i, round(err[i, j], 1), ha="center", va="center", color="w")
    im = ax.imshow(err)
    ax.set_title("Dupire Calibration PV Error CS_IV_BS - AF_LV_PDE - Matrix")
    fig.tight_layout()
    plt.show()

def pdeCalibReport_New2(S0, r, q, impliedVol):
    report_start = time_block_start()
    
    ts = [0.02, 0.04, 0.06, 1/12.0, 1/6.0, 1/4.0, 1/2.0, 1, 2, 5]
    ds = np.arange(0.1, 1.0, 0.1)
    
    lv = LocalVol(impliedVol, S0, r, q)
    
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(ts)))
    ax.set_xlabel("T")
    ax.set_yticks(np.arange(len(ds)))
    ax.set_ylabel("Put Delta")
    ax.set_xticklabels(map(lambda t : round(t, 2), ts))
    ax.set_yticklabels(map(lambda d : round(d, 1), ds))
    
    def compute_error(i, j):
        T = ts[j]
        K = strikeFromDelta(S0, r, 0, T, impliedVol.Vol(T, S0*math.exp(r*T)), ds[i], PayoffType.Put)
        payoff = PayoffType.Put
        trade = EuropeanOption("ASSET1", T, K, payoff)
        vol = impliedVol.Vol(ts[j], K)
        bs = bsPrice(S0, r, q, vol, T, K, payoff)
        pde = pdePricerX(S0, r, q, lv, max(50, int(50 * T)), max(50, int(50 * T)), 0.5, trade)
        return math.fabs(bs - pde)/S0 * 10000

    local_vol_start = time_block_start()
    errors = Parallel(n_jobs=-1)(delayed(compute_error)(i, j) for i in range(len(ds)) for j in range(len(ts)))
    err = np.array(errors).reshape(len(ds), len(ts))
    time_block_end(local_vol_start, "Local Vol Surface Creation")
    
    cax = ax.imshow(err)
    ax.set_title("Dupire Calibration PV Error Matrix using optimized function")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(ds)):
        for j in range(len(ts)):
            ax.text(j, i, f'{err[i, j]:.2f}', ha='center', va='center', color='w')
    
    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    plt.show()
    
    time_block_end(report_start, "Total Calibration Report")

def time_block_start():
    return time.time()

def time_block_end(start, description):
    elapsed = time.time() - start
    print(f"Time taken for '{description}': {elapsed:.4f} seconds")