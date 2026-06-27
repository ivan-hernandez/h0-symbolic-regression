"""
Download core microbial metabolic scaling datasets.
All sources are open access.
"""
import os, sys, urllib.request, zipfile, gzip, shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"  Already exists: {filename}")
        return path
    print(f"  Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  Saved {filename} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print(f"  FAILED: {e}")
    return path

def main():
    # --- 1. DeLong+2010 PNAS supplementary data ---
    # Datasets S1 and S2 (by mass, by taxon)
    print("\n[1] DeLong+2010 (PNAS)")
    download(
        "https://www.pnas.org/doi/suppl/10.1073/pnas.1007783107/-/DCSupplemental/sd01.doc",
        "delong2010_dataset_s1.doc"
    )
    download(
        "https://www.pnas.org/doi/suppl/10.1073/pnas.1007783107/-/DCSupplemental/sd02.doc",
        "delong2010_dataset_s2.doc"
    )

    # --- 2. Makarieva+2005 supplementary data ---
    print("\n[2] Makarieva+2005 (Proc B)")
    download(
        "https://royalsocietypublishing.org/doi/suppl/10.1098/rspb.2005.3225/suppl_file/rspb20053225supp1.pdf",
        "makarieva2005_supplementary.pdf"
    )

    # --- 3. Hoehler+2023 Zenodo ---
    print("\n[3] Hoehler+2023 (PNAS, Zenodo)")
    # Try the interactive HTML first, then look for CSV/XLSX
    download(
        "https://zenodo.org/records/7877885/files/mtab_interactive_nomad.html?download=1",
        "hoehler2023_interactive.html"
    )
    # There should be a data file on Zenodo
    download(
        "https://zenodo.org/records/7877885/files/mtab_data.csv?download=1",
        "hoehler2023_data.csv"
    )
    # Also try xlsx
    download(
        "https://zenodo.org/records/7877885/files/mtab_data.xlsx?download=1",
        "hoehler2023_data.xlsx"
    )

    # --- 4. Kiørboe+Hirst 2013 PANGAEA ---
    print("\n[4] Kiørboe+Hirst 2013 (PANGAEA)")
    # Try the tab-separated text export
    download(
        "https://doi.pangaea.de/10.1594/PANGAEA.819857?format=textfile",
        "kiørboe2013.txt"
    )

    # --- 5. Also try Kempes+2012 ---
    print("\n[5] Kempes+2012 (PNAS)")
    download(
        "https://www.pnas.org/doi/suppl/10.1073/pnas.1209429109/-/DCSupplemental/sd01.xls",
        "kempes2012_sd01.xls"
    )

    print("\nDone. Check data/ directory for downloaded files.")


if __name__ == '__main__':
    main()
