"""Parse SPARC mass models and RAR data."""
import numpy as np
import pandas as pd

TABLE_PATH = "MassModels_Lelli2016c.mrt"

def parse_mass_models(path=TABLE_PATH):
    """Parse the SPARC mass model table (fixed-width MRT format)."""
    with open(path) as f:
        lines = f.readlines()

    # Data starts after the last --- separator.
    separator_lines = [i for i, line in enumerate(lines) if line.strip().startswith("---")]
    if not separator_lines:
        raise ValueError("Could not find data start (no separator lines)")
    data_start = separator_lines[-1] + 1
    print(f"Data starts at line {data_start + 1}")

    # Fixed-width columns from MRT spec:
    # ID: [0:11], D: [12:18], R: [19:25], Vobs: [26:32],
    # e_Vobs: [33:38], Vgas: [39:45], Vdisk: [46:52],
    # Vbul: [53:59], SBdisk: [60:67], SBbul: [68:76]
    cols = [
        (0, 11, "ID", str),
        (12, 18, "D", float),
        (19, 25, "R", float),
        (26, 32, "Vobs", float),
        (33, 38, "e_Vobs", float),
        (39, 45, "Vgas", float),
        (46, 52, "Vdisk", float),
        (53, 59, "Vbul", float),
        (60, 67, "SBdisk", float),
        (68, 76, "SBbul", float),
    ]

    rows = []
    for line in lines[data_start:]:
        if len(line.strip()) < 20:
            continue
        row = {}
        for start, end, name, dtype in cols:
            val = line[start:end].strip()
            if dtype == float:
                try:
                    row[name] = float(val) if val else 0.0
                except ValueError:
                    row[name] = 0.0
            else:
                row[name] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} rows, {df['ID'].nunique()} galaxies")
    print(f"Columns: {list(df.columns)}")

    # Verify ranges
    print(f"  R range: [{df['R'].min():.2f}, {df['R'].max():.2f}] kpc")
    print(f"  Vobs range: [{df['Vobs'].min():.1f}, {df['Vobs'].max():.1f}] km/s")
    print(f"  z range: [{df['D'].min():.1f}, {df['D'].max():.1f}] Mpc")
    return df


def compute_radial_accelerations(df, Upsilon_disk=0.5, Upsilon_bul=0.7, use_full_err=True):
    """Drop center points where R=0 to avoid division by zero."""
    df = df[df["R"] > 0].copy()
    """Compute gbar and gobs from rotation curve data.

    gbar = Vbar^2 / R  where Vbar^2 = Vgas*|Vgas| + Upsilon_disk*Vdisk^2 + Upsilon_bul*Vbul^2
    gobs = Vobs^2 / R

    Returns units of m/s^2 (converted from km/s and kpc).
    """
    kpc_to_m = 3.0857e19
    km_s_to_m_s = 1000.0

    Vbar_sq = (
        np.abs(df["Vgas"].values) * df["Vgas"].values
        + Upsilon_disk * df["Vdisk"].values ** 2
        + Upsilon_bul * df["Vbul"].values ** 2
    )
    # Avoid negative Vbar_sq (can happen with small numbers)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    Vbar = np.sqrt(Vbar_sq)

    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * km_s_to_m_s ** 2 / R_m  # m/s^2
    gobs = df["Vobs"].values ** 2 * km_s_to_m_s ** 2 / R_m

    if use_full_err and "e_Vobs" in df.columns:
        dgobs_dVobs = 2 * df["Vobs"].values * km_s_to_m_s ** 2 / R_m
        e_gobs = np.abs(dgobs_dVobs) * df["e_Vobs"].values
    else:
        e_gobs = np.full_like(gobs, np.nan)

    result = pd.DataFrame({
        "ID": df["ID"],
        "D": df["D"],
        "R": df["R"],
        "Vobs": df["Vobs"],
        "gbar": gbar,
        "gobs": gobs,
        "e_gobs": e_gobs,
        "log_gbar": np.log10(gbar),
        "log_gobs": np.log10(gobs),
    })
    print(f"gbar: [{np.log10(gbar.min()):.2f}, {np.log10(gbar.max()):.2f}]")
    print(f"gobs: [{np.log10(gobs.min()):.2f}, {np.log10(gobs.max()):.2f}]")
    return result


def load_rar(path="RAR.mrt"):
    """Load the pre-computed RAR data."""
    df = pd.read_csv(path, comment="#", delim_whitespace=True, skiprows=2,
                     names=["gbar", "e_gbar", "gobs", "e_gobs"])
    print(f"Loaded RAR: {len(df)} points")
    print(f"  log_gbar: [{df['gbar'].min():.2f}, {df['gbar'].max():.2f}]")
    print(f"  log_gobs: [{df['gobs'].min():.2f}, {df['gobs'].max():.2f}]")
    return df


if __name__ == "__main__":
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    print(f"\nExample galaxy ({acc['ID'].iloc[0]}):")
    print(acc.head())
    print(f"\nTotal points: {len(acc)}")
    print(f"Unique galaxies: {acc['ID'].nunique()}")
