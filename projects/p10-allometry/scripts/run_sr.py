"""
Phase 1: PySR discovery on metabolic allometry.
Run with: python run_sr.py --seed N --subset all|endotherm|ectotherm|mammalia|aves
"""
import argparse, os, sys, pickle
import numpy as np
from pysr import PySRRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data.npz')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

SUBSETS = ['all', 'endotherm', 'ectotherm', 'mammalia', 'mammal_bmr', 'aves', 'insecta']

def load_subset(subset):
    d = np.load(DATA_PATH)
    arr = d['data']
    if subset == 'all':
        mask = np.ones(len(arr), dtype=bool)
    elif subset == 'endotherm':
        mask = np.array([r['thermo'] == 'endotherm' for r in arr], dtype=bool)
    elif subset == 'ectotherm':
        mask = np.array([r['thermo'] == 'ectotherm' for r in arr], dtype=bool)
    elif subset == 'mammalia':
        mask = np.array([r['class'] == 'Mammalia' for r in arr], dtype=bool)
    elif subset == 'mammal_bmr':
        mask = np.array([r['class'] == 'Mammalia' and r['method'] == 'basal metabolic rate' for r in arr], dtype=bool)
    elif subset == 'aves':
        mask = np.array([r['class'] == 'Aves' for r in arr], dtype=bool)
    elif subset == 'insecta':
        mask = np.array([r['class'] == 'Insecta' for r in arr], dtype=bool)
    else:
        raise ValueError(f"Unknown subset: {subset}")
    log_mass = arr['log_mass'][mask]
    log_mr = arr['log_mr'][mask]
    print(f"Subset '{subset}': {len(log_mass)} points, "
          f"mass range {10**log_mass.min():.2e} - {10**log_mass.max():.2e} kg")
    return log_mass.reshape(-1, 1), log_mr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subset', choices=SUBSETS, default='all')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--populations', type=int, default=20)
    args = parser.parse_args()

    X, y = load_subset(args.subset)

    model = PySRRegressor(
        niterations=args.iterations,
        populations=args.populations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "square", "cube"],
        maxsize=30,
        parsimony=0.001,
        model_selection="accuracy",
        elementwise_loss="L2DistLoss()",
        denoise=False,
        warm_start=False,
        verbosity=1,
        random_state=args.seed,
        procs=12,
        precision=32,
        constraints={"exp": 3, "log": 2, "sqrt": 2, "square": 2, "cube": 2},
        nested_constraints={
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0, "sqrt": 0},
            "sqrt": {"exp": 0, "log": 0, "sqrt": 0},
        },
    )

    print(f"\nStarting PySR (seed={args.seed}, subset={args.subset}, "
          f"iter={args.iterations}, pop={args.populations})")
    model.fit(X, y)

    # Save model
    outfile = os.path.join(OUT_DIR, f"model_{args.subset}_s{args.seed}.pkl")
    with open(outfile, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved {outfile}")

    # Print equations
    print("\n" + "="*60)
    print(f"Equations for {args.subset} (seed={args.seed})")
    print("="*60)
    df = model.equations_
    for idx, row in df.iterrows():
        score = row.get('score', 0) or 0
        print(f"CPX{row['complexity']:2d} | score={score:.4f} | loss={row['loss']:.4f} | {row['equation']}")

    # Save equation summary
    summary_path = os.path.join(OUT_DIR, f"equations_{args.subset}_s{args.seed}.txt")
    df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

if __name__ == "__main__":
    main()
