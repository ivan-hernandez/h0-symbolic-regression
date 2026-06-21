"""Shared data loader: CC, SDSS BAO, DESI DR1/DR2 BAO, Pantheon+."""
import numpy as np, urllib.request, os

C = 299792.458

# ---------- Cosmic Chronometers ----------
def get_cc():
    return np.array([
        [0.070,69.0,19.6],[0.090,69.0,12.0],[0.120,68.6,26.2],
        [0.170,83.0,8.0],[0.1791,75.0,4.0],[0.1993,75.0,5.0],
        [0.200,72.9,29.6],[0.270,77.0,14.0],[0.280,88.8,36.6],
        [0.3519,83.0,14.0],[0.3802,83.0,13.5],[0.400,95.0,17.0],
        [0.4004,77.0,10.2],[0.4247,87.1,11.2],[0.4497,92.8,12.9],
        [0.470,89.0,34.0],[0.4783,80.9,9.0],[0.480,97.0,62.0],
        [0.5929,104.0,13.0],[0.6797,92.0,8.0],[0.750,98.8,33.6],
        [0.7812,105.0,12.0],[0.800,113.1,28.5],[0.8754,125.0,17.0],
        [0.880,90.0,40.0],[0.900,117.0,23.0],[1.037,154.0,20.0],
        [1.300,168.0,17.0],[1.363,160.0,33.6],[1.430,177.0,18.0],
        [1.530,140.0,14.0],[1.750,202.0,40.0],[1.965,186.5,50.4],
    ])

# ---------- SDSS BAO (Alam+ 2017) ----------
def get_bao_sdss():
    return np.array([
        [0.380,81.1,2.2],[0.510,91.1,2.1],[0.610,99.4,2.2],
    ])

# ---------- DESI DR1 BAO ----------
def get_desi_dr1(r_s=147.0):
    pts = np.array([
        [0.510,20.98334647,0.61],[0.706,20.07872919,0.60],
        [0.930,17.87612922,0.35],[1.317,13.82372285,0.42],
        [2.330,8.52256583,0.17],
    ])
    hz = C/(r_s*pts[:,1]); err = hz*pts[:,2]/pts[:,1]
    return np.column_stack([pts[:,0],hz,err])

# ---------- DESI DR2 BAO (arXiv:2503.14738) ----------
def get_desi_dr2(r_s=147.0):
    """D_H/r_s measurements from DESI DR2 combined galaxy+quasar sample."""
    pts = np.array([
        [0.510,21.86294686,0.42892],
        [0.706,19.45534918,0.33387],
        [0.934,17.64149464,0.20104],
        [1.321,14.17602155,0.22455],
        [1.484,12.81699964,0.51801],
        [2.330,8.63154567,0.10106],
    ])
    hz = C/(r_s*pts[:,1]); err = hz*pts[:,2]/pts[:,1]
    return np.column_stack([pts[:,0],hz,err])

# ---------- Combined data ----------
def load_hz(include_sdss=True, version='dr2', r_s=147.0):
    cc = get_cc(); bao = get_bao_sdss()
    combined = np.vstack([cc, bao])
    if version == 'dr1':
        desi = get_desi_dr1(r_s)
    else:
        desi = get_desi_dr2(r_s)
    combined = np.vstack([combined, desi])
    combined = combined[combined[:,0].argsort()]
    mask = (combined[:,2]>0)&(combined[:,1]>0)&(combined[:,2]<100)
    return combined[mask]

# ---------- Pantheon+ ----------
PANTHEON_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")

def fetch_pantheon():
    req = urllib.request.urlopen(PANTHEON_URL, timeout=30)
    lines = req.read().decode().strip().split('\n')
    hdr = lines[0].split()
    zi = hdr.index('zHD'); mi = hdr.index('MU_SH0ES')
    ei = hdr.index('MU_SH0ES_ERR_DIAG')
    z,m,e = [],[],[]
    for line in lines[1:]:
        c = line.split()
        if len(c) <= max(zi,mi,ei): continue
        zh,mu,err = float(c[zi]),float(c[mi]),float(c[ei])
        if zh>0.01 and err>0: z.append(zh); m.append(mu); e.append(err)
    return np.array(z),np.array(m),np.array(e)

# ---------- Simpson integration ----------
def quad_simpson(f, a, b, n=2000):
    xs = np.linspace(a, b, 2*n+1)
    h = (b-a)/(2*n)
    fx = f(xs)
    return h/3*(fx[0]+fx[-1]+4*np.sum(fx[1::2])+2*np.sum(fx[2:-1:2]))

def mu_from_H(H_func, z):
    Dc = C * quad_simpson(lambda zp: 1.0/H_func(zp), 0, z)
    return 5.0*np.log10((1+z)*Dc) + 25.0
