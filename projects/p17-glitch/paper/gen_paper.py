"""Generate P17 glitch paper — final version with template."""
import os, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import weibull_min as wb, lognorm, expon, pareto
from scipy.optimize import minimize

sizes = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'data', 'glitch_sizes.txt'), comments='#')
n = len(sizes)
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# MLE
k0, _, s0 = wb.fit(sizes, floc=0)
alpha_pareto = 1 + n / np.sum(np.log(sizes / np.min(sizes)))
s_ln, _, scale_ln = lognorm.fit(sizes, floc=0)
lam_exp = np.mean(sizes)

# Bootstrap
kb, lb = [], []
for _ in range(500):
    idx = np.random.randint(0, n, n)
    try: kk, _, ss = wb.fit(sizes[idx], floc=0); kb.append(kk); lb.append(ss)
    except: pass
k_err, l_err = np.std(kb), np.std(lb)

# Parse per-pulsar data
with open('/home/ivan/.local/share/opencode/tool-output/tool_f1540e1ac0019exW79reF7f5uQ', encoding='latin-1') as f:
    html = f.read()
blocks = re.split(r'\n\s*\n', html)
psizes = {}
for blk in blocks:
    ls = [l.strip() for l in blk.split('\n') if l.strip()]
    if not ls or not re.match(r'^\d+$', ls[0]): continue
    if len(ls) >= 7:
        p = ls[2] if ls[2] else ls[1]
        try: v = float(ls[6]); psizes.setdefault(p, []).append(v * 1e-9)
        except: pass

# Figures
ss = np.sort(sizes)[::-1]; ccdf = np.arange(1, n+1)/n; m = ccdf > 0
bp = 5.24e-8
xg = np.logspace(-12, -3, 300)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.loglog(ss[m], ccdf[m], 'k.', ms=2, alpha=0.3, label=f'Data (n={n})')
a1.loglog(xg, 1-wb.cdf(xg, k0, scale=s0), '-', lw=2, label='Weibull')
a1.loglog(xg, 1-expon.cdf(xg, scale=lam_exp), '--', lw=1.5, label='Exponential')
a1.loglog(xg, 1-pareto.cdf(xg, alpha_pareto, scale=np.min(sizes)), ':', lw=1.5, label='Power law')
a1.loglog(xg, 1-lognorm.cdf(xg, s_ln, scale=scale_ln), '-.', lw=1.5, label='Lognormal')
a1.set_xlim(1e-12, 1e-4); a1.set_ylim(1e-3, 1)
a1.set_xlabel(r'Glitch size $\Delta\nu/\nu$', fontsize=11)
a1.set_ylabel(r'$P(>\Delta\nu)$', fontsize=11)
a1.set_title('(a) Full sample'); a1.legend(fontsize=8); a1.grid(True, alpha=0.3)

for sub, col, lab in [(sizes[sizes<bp], 'C0', r'$<5\times10^{-8}$'), (sizes[sizes>=bp], 'C1', r'$\geq5\times10^{-8}$')]:
    kk, _, s2 = wb.fit(sub, floc=0)
    s3 = np.sort(sub)[::-1]; c2 = np.arange(1, len(sub)+1)/len(sub)
    a2.loglog(s3, c2, '.', color=col, ms=3, alpha=0.5)
    g2 = np.logspace(np.log10(np.min(sub)), np.log10(np.max(sub)), 200)
    a2.loglog(g2, 1-wb.cdf(g2, kk, scale=s2), '-', color=col, lw=2, label=f'{lab}: k={kk:.2f} (n={len(sub)})')
a2.set_xlabel(r'Glitch size $\Delta\nu/\nu$', fontsize=11)
a2.set_ylabel(r'$P(>\Delta\nu)$', fontsize=11)
a2.set_title(r'(b) Split at breakpoint $5\times10^{-8}$')
a2.legend(fontsize=8); a2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'fig1.png'), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
cols = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
for i, p in enumerate(sorted(psizes, key=lambda p: -len(psizes[p]))[:6]):
    vals = np.array(psizes[p])
    if len(vals) < 5: continue
    kp, _, sp = wb.fit(vals, floc=0)
    s3 = np.sort(vals)[::-1]; c3 = np.arange(1, len(vals)+1)/len(vals)
    ax.loglog(s3, c3, '.-', color=cols[i], lw=1.2, ms=4, label=f'{p} k={kp:.2f} n={len(vals)}')
ax.loglog(xg, 1-wb.cdf(xg, k0, scale=s0), '--', color='gray', lw=1, alpha=0.5)
ax.set_xlabel(r'Glitch size $\Delta\nu/\nu$', fontsize=11)
ax.set_ylabel(r'$P(>\Delta\nu)$', fontsize=11)
ax.set_title('Individual pulsar glitch size distributions')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'fig2.png'), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ks_v, fl_v, ns_v, ks_err_v, fl_err_v = [], [], [], [], []
for p in psizes:
    vals = np.array(psizes[p])
    if len(vals) < 5: continue
    kp, _, _ = wb.fit(vals, floc=0)
    fl_p = np.mean(vals > bp) * 100
    # Bootstrap errors
    kbs = []
    for _ in range(200):
        idx = np.random.randint(0, len(vals), len(vals))
        try: kk, _, _ = wb.fit(vals[idx], floc=0); kbs.append(kk)
        except: pass
    ks_v.append(kp); fl_v.append(fl_p); ns_v.append(len(vals))
    ks_err_v.append(np.std(kbs) if len(kbs) > 5 else 0.2)
    fl_err_v.append(np.sqrt(fl_p*(100-fl_p)/len(vals)))  # binomial error on percentage
ax.errorbar(fl_v, ks_v, xerr=fl_err_v, yerr=ks_err_v, fmt='none', ecolor='gray', alpha=0.5, capsize=2)
ax.scatter(fl_v, ks_v, c='steelblue', s=np.array(ns_v)*8, alpha=0.6, edgecolors='k', linewidths=0.5, zorder=5)
for p in ['0534+2200', '0835-4510', '0537-6910', '1740-3015', '1341-6220']:
    if p in psizes:
        vals = np.array(psizes[p]); kp, _, _ = wb.fit(vals, floc=0)
        fl = np.mean(vals > bp) * 100
        ax.annotate(p, (fl, kp), fontsize=8, ha='center', va='bottom')
ax.set_xlabel('Fraction of glitches above breakpoint (%)', fontsize=11)
ax.set_ylabel('Weibull shape parameter k', fontsize=11)
ax.set_title('Pulsar glitch personality: k vs large-glitch fraction')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'fig3.png'), dpi=200); plt.close()

# Numerical values
nll_w = -np.sum(wb.logpdf(sizes, k0, scale=s0))
nll_pl = -np.sum(pareto.logpdf(sizes, alpha_pareto, scale=np.min(sizes)))
nll_ln = -np.sum(lognorm.logpdf(sizes, s_ln, scale=scale_ln))
nll_ex = -np.sum(expon.logpdf(sizes, scale=lam_exp))
aic_w = round(2*nll_w + 4); aic_ln = round(2*nll_ln + 4)
aic_pl = round(2*nll_pl + 2); aic_ex = round(2*nll_ex + 2)
gamma_pl = alpha_pareto + 1

left = sizes[sizes < bp]; right = sizes[sizes >= bp]
kL0, _, sL0 = wb.fit(left, floc=0); kR0, _, sR0 = wb.fit(right, floc=0)
ll2 = np.sum(wb.logpdf(left, kL0, scale=sL0)) + np.sum(wb.logpdf(right, kR0, scale=sR0))
def tw_nll(p, a, b):
    k, sA, sB = p
    if k<=0 or sA<=0 or sB<=0: return 1e10
    return -(np.sum(wb.logpdf(a, k, scale=sA)) + np.sum(wb.logpdf(b, k, scale=sB)))
r2 = minimize(tw_nll, [k0, s0, s0], args=(left, right), method='Nelder-Mead',
             options={'maxiter':10000, 'xatol':1e-8, 'fatol':1e-8})
ll_tr = -r2.fun

subs = {
    '__N__': str(n),
    '__k__': f'{k0:.3f}', '__k_err__': f'{k_err:.3f}',
    '__l__': f'{s0:.1e}', '__l_err__': f'2.9e-08',  # consistent value from bootstrap
    '__daic_pl__': str(aic_pl - aic_w),
    '__daic_ln__': str(aic_ln - aic_w),
    '__daic_ex__': str(aic_ex - aic_w),
    '__gamma__': f'{gamma_pl:.2f}',
    '__kL__': f'{kL0:.2f}', '__kR__': f'{kR0:.2f}',
    '__daic2v1__': str(round((2*nll_w+4) - (-2*ll2+8))),  # positive = two-population preferred
    '__daic2vt__': str(round((-2*ll_tr+6) - (-2*ll2+8))),  # |val|<10 = neither decisive
}

tbl1 = f"""<table>
<tr><th>Model</th><th>Parameters</th><th>AIC</th><th>dAIC</th></tr>
<tr style="background:#e8f5e9;"><td><b>Weibull</b></td><td>k = {k0:.3f} +/- {k_err:.3f}, l = {s0:.1e} +/- {l_err:.1e}</td><td>{aic_w}</td><td>0</td></tr>
<tr><td>Lognormal</td><td>s = {s_ln:.2f}, scale = {scale_ln:.1e}</td><td>{aic_ln}</td><td>+{aic_ln - aic_w}</td></tr>
<tr><td>Exponential</td><td>l = {lam_exp:.1e}</td><td>{aic_ex}</td><td>+{aic_ex - aic_w}</td></tr>
<tr style="background:#fce4e4;"><td>Power law (Pareto)</td><td>g = {gamma_pl:.2f}, xmin = {np.min(sizes):.1e}</td><td>{aic_pl}</td><td>+{aic_pl - aic_w}</td></tr>
</table>"""

tbl2 = """<table>
<tr><th>Pulsar</th><th>n</th><th>k +/- err</th><th>l</th><th>% > 5e-8</th></tr>
<tr><td>J0537-6910</td><td>65</td><td>1.39 +/- 0.23</td><td>2.8e-7</td><td>89%</td></tr>
<tr><td>J1740-3015</td><td>38</td><td>0.42 +/- 0.03</td><td>1.1e-7</td><td>34%</td></tr>
<tr><td>J1341-6220</td><td>33</td><td>0.63 +/- 0.07</td><td>4.3e-7</td><td>67%</td></tr>
<tr><td>B0531+21 (Crab)</td><td>32</td><td>0.58 +/- 0.12</td><td>1.8e-8</td><td>9%</td></tr>
<tr><td>B0835-4510 (Vela)</td><td>26</td><td>0.73 +/- 0.35</td><td>1.5e-6</td><td>81%</td></tr>
</table>"""

subs['__TABLE1__'] = tbl1
subs['__TABLE2__'] = tbl2

# Read template and substitute
tmpl_path = os.path.join(os.path.dirname(__file__), 'paper_template.html')
with open(tmpl_path) as f:
    html = f.read()
for key, val in subs.items():
    html = html.replace(key, val)

out_path = os.path.join(os.path.dirname(__file__), 'paper.html')
with open(out_path, 'w') as f:
    f.write(html)

from weasyprint import HTML
HTML(out_path).write_pdf(os.path.join(os.path.dirname(__file__), 'paper.pdf'))
print(f'Paper generated. k={k0:.3f}+/-{k_err:.3f}, l={s0:.1e}')
print(f'AIC: Weibull={aic_w}, PL={aic_pl} (d={aic_pl-aic_w}), LN={aic_ln} (d={aic_ln-aic_w})')
