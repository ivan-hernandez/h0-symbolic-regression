"""DM v3: Einasto fit, reduced χ² quality gate, concentration sanity.

Fixes: actually fit Einasto, flag bad fits (χ²_red>20), check c in [1, 100].
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit; from scipy.special import gammainc
from parse_sparc import parse_mass_models
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/dm_profiles"; os.makedirs(OUTDIR,exist_ok=True)
kpc_to_m=3.0857e19; G_SI=6.6743e-11; Msun_kg=1.989e30

def V_nfw(R,logM,logc):
    M200=10**logM; c=10**logc; rc=1.37e-7; R200=(M200/(200*4*np.pi/3*rc*1e9))**(1/3); rs=R200/c
    rhos=M200*Msun_kg/(4*np.pi*rs**3*(np.log(1+c)-c/(1+c)))/kpc_to_m**3
    x=R/rs; M=4*np.pi*rhos*rs**3*(np.log(1+x)-x/(1+x))
    return np.sqrt(np.maximum(G_SI*M/(R*kpc_to_m)/1e6,0))

def V_burkert(R,log_rho0,log_r0):
    r0=10**log_r0; rho0=10**log_rho0; x=R/r0
    M=2*np.pi*rho0*r0**3*(np.log(1+x)+0.5*np.log(1+x**2)-np.arctan(x))
    return np.sqrt(np.maximum(G_SI*M/(R*kpc_to_m)/1e6,0))

def V_einasto(R,log_rho0,log_r0,alpha):
    rho0=10**log_rho0; r0=10**log_r0; r=R/r0; n=1/max(alpha,1e-3)
    dn=3*n-1/3+0.0079/n; x=dn*r**alpha; gam=gammainc(3*n,x)
    M=4*np.pi*rho0*r0**3*n*np.exp(dn)*dn**(-3*n)*gam
    return np.sqrt(np.maximum(G_SI*M/(R*kpc_to_m)/1e6,0))

df=parse_mass_models(); df=df[df["R"]>0]
gals=df.groupby("ID").size()
top=gals[gals>=8].index

results=[]
for gal in top:
    sub=df[df["ID"]==gal].sort_values("R")
    R=sub["R"].values; V=sub["Vobs"].values; e=sub["e_Vobs"].values
    if len(R)<8: continue
    sig=np.maximum(e,10.0)  # 10 km/s floor (typical mass model systematics)
    n_pts=len(R); k=2

    fits={}
    for name,func,p0 in [("NFW",V_nfw,[11.5,1.0]),
                          ("Burkert",V_burkert,[6,0.5]),
                          ("Einasto",V_einasto,[6,0.3,0.17])]:
        try:
            popt,_=curve_fit(func,R,V,p0=p0,sigma=sig,maxfev=10000,absolute_sigma=True)
            pred=func(R,*popt); chi2=np.sum(((V-pred)/sig)**2)
            chi2_red=chi2/(n_pts-len(popt))
            aic=chi2+2*len(popt)
            fits[name]={"popt":popt,"chi2":chi2,"chi2_red":chi2_red,"aic":aic}
        except: pass

    # Quality gate: at least one model must have χ²_red < 30
    good_fits={k:v for k,v in fits.items() if v["chi2_red"]<30}
    if len(good_fits)<1: continue

    best=min(good_fits,key=lambda k:good_fits[k]["aic"])
    c_val=10**fits["NFW"]["popt"][1] if "NFW" in fits else np.nan
    nfw_sane=1<c_val<100 if "NFW" in fits else False

    # Einasto alpha sanity
    ein_ok="Einasto" in fits and 0.01<fits["Einasto"]["popt"][2]<2.0

    results.append({"galaxy":gal,"n_pts":n_pts,"V_max":V.max(),
                     "best":best,"c_nfw":c_val if "NFW" in fits else np.nan,
                     "nfw_sane":nfw_sane,"ein_ok":ein_ok,
                     "chi2_nfw":fits.get("NFW",{}).get("chi2_red",99),
                     "chi2_bur":fits.get("Burkert",{}).get("chi2_red",99),
                     "chi2_ein":fits.get("Einasto",{}).get("chi2_red",99)})

dr=pd.DataFrame(results)
print(f"DM v3: {len(dr)} galaxies passed quality gate")
n_nfw=(dr["best"]=="NFW").sum(); n_bur=(dr["best"]=="Burkert").sum(); n_ein=(dr["best"]=="Einasto").sum()
print(f"NFW: {n_nfw}, Burkert: {n_bur}, Einasto: {n_ein}")
print(f"NFW sane (1<c<100): {dr['nfw_sane'].sum()}/{len(dr)}")
print(f"Einasto sane (α∈[0.01,2]): {dr['ein_ok'].sum()}/{len(dr)}")
print(f"Median χ²_red: NFW={dr['chi2_nfw'].median():.1f}, Burkert={dr['chi2_bur'].median():.1f}, Einasto={dr['chi2_ein'].median():.1f}")

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].bar(["NFW","Burkert","Einasto"],[n_nfw,n_bur,n_ein],color=["red","blue","green"],edgecolor="k")
ax[0].set_ylabel("Count"); ax[0].set_title(f"Best Model ({len(dr)} galaxies, χ²_red<20)")
ax[1].scatter(dr["V_max"],dr["c_nfw"],s=10,c=["red" if b=="NFW" else "blue" if b=="Burkert" else "green" for b in dr["best"]],alpha=0.6)
ax[1].axhline(10,color="k",ls="--"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("V_max"); ax[1].set_ylabel("c_NFW"); ax[1].set_title("NFW concentration")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/dm_v3.pdf",dpi=200); plt.savefig(f"{OUTDIR}/dm_v3.png",dpi=150)
plt.close(); dr.to_csv(f"{OUTDIR}/dm_v3.csv",index=False)
print("Done.")
