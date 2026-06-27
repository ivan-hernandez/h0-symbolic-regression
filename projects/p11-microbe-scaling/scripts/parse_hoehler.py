"""
Extract all ColumnDataSource data from Hoehler+2023 Bokeh plot.
"""
import json, re, os, csv

HTML_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hoehler2023_interactive.html')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    with open(HTML_PATH) as f:
        html = f.read()

    match = re.search(r'<script type="application/json"[^>]*id="3718"[^>]*>(.*?)</script>', html, re.DOTALL)
    doc = json.loads(match.group(1))
    root_id = list(doc.keys())[0]
    refs = doc[root_id]['roots']['references']

    datasets = {}
    for ref in refs:
        if ref.get('type') == 'ColumnDataSource':
            data = ref['attributes'].get('data', {})
            if data:
                nrows = len(list(data.values())[0])
                name = ref['attributes'].get('name', 'unnamed')
                key = f"{name}_{nrows}rows"
                datasets[key] = {'name': name, 'nrows': nrows, 'data': data}

    for key, ds in sorted(datasets.items()):
        n = ds['nrows']
        cols = list(ds['data'].keys())
        print(f"\n=== {key} ===")
        print(f"  Columns ({len(cols)}): {cols}")
        
        if n > 0 and n < 10000:
            # Save as CSV
            csv_path = os.path.join(OUT_DIR, f"hoehler_{key}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for i in range(n):
                    row = [ds['data'][c][i] if i < len(ds['data'][c]) else '' for c in cols]
                    writer.writerow(row)
            print(f"  Saved to {csv_path}")

    print(f"\nDone. {sum(1 for k in datasets if datasets[k]['nrows'] > 0)} non-empty datasets extracted.")

if __name__ == '__main__':
    main()
