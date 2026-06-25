"""One-click publish: Zenodo + OSF + GitHub in a single command.

Usage: python3 publish.py "Paper Title" "Description" paper_dir/

Sets:
- ZENODO_TOKEN, OSF_TOKEN as env vars or hard-code below
- Creates Zenodo deposition, uploads PDF+TeX+figures, publishes
- Creates OSF component with DOI link
- Prints Zenodo DOI and OSF URL
"""
import os, sys, json, subprocess

# ── Config (set via env vars; never hardcode tokens) ──
ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN")
OSF_TOKEN = os.environ.get("OSF_TOKEN")
if not ZENODO_TOKEN or not OSF_TOKEN:
    sys.exit("ERROR: Set ZENODO_TOKEN and OSF_TOKEN env vars")
OSF_PARENT = "j9a7u"  # Parent OSF project
GITHUB_REPO = "https://github.com/ivan-hernandez/h0-symbolic-regression"

# ── CLI ──
if len(sys.argv) < 3:
    print("Usage: python3 publish.py 'Title' 'Description' paper_dir/")
    print("  paper_dir/ should contain: paper.pdf, paper.tex")
    sys.exit(1)

TITLE = sys.argv[1]
DESC = sys.argv[2]
PAPER_DIR = sys.argv[3] if len(sys.argv) > 3 else "."

import urllib.request, base64

def zenodo_request(method, url, data=None, content_type="application/json", binary_data=None):
    headers = {"Authorization": f"Bearer {ZENODO_TOKEN}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def osf_request(method, url, data=None):
    headers = {"Authorization": f"Bearer {OSF_TOKEN}", "Content-Type": "application/vnd.api+json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

# ── 1. Create Zenodo deposition ──
print("=== 1. Creating Zenodo deposition ===")
deposit = zenodo_request("POST", "https://zenodo.org/api/deposit/depositions",
    json.dumps({
        "metadata": {
            "title": TITLE,
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": DESC,
            "creators": [{"name": "Hernandez, Ivan"}],
            "access_right": "open",
            "license": "MIT",
            "related_identifiers": [{"relation": "isSupplementedBy", "identifier": GITHUB_REPO}]
        }
    }).encode())

dep_id = deposit["id"]
bucket = deposit["links"]["bucket"]
pre_doi = deposit["metadata"]["prereserve_doi"]["doi"]
print(f"  ID: {dep_id}, Pre-DOI: {pre_doi}")

# ── 2. Upload files ──
print("=== 2. Uploading files ===")
for filename in os.listdir(PAPER_DIR):
    filepath = os.path.join(PAPER_DIR, filename)
    if not os.path.isfile(filepath):
        continue
    if not (filename.endswith(".pdf") or filename.endswith(".tex") or
            filename.endswith(".png") or filename.endswith(".csv")):
        continue
    with open(filepath, "rb") as f:
        data = f.read()
    headers = {"Authorization": f"Bearer {ZENODO_TOKEN}", "Content-Type": "application/octet-stream"}
    req = urllib.request.Request(f"{bucket}/{filename}", data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=30) as resp:
        pass
    print(f"  Uploaded: {filename} ({len(data)} bytes)")

# ── 3. Publish ──
print("=== 3. Publishing ===")
pub = zenodo_request("POST", f"https://zenodo.org/api/deposit/depositions/{dep_id}/actions/publish")
doi = pub.get("doi", pub.get("metadata", {}).get("prereserve_doi", {}).get("doi", pre_doi))
print(f"  Published: https://doi.org/{doi}")

# ── 4. Create OSF component ──
print("=== 4. Creating OSF component ===")
comp = osf_request("POST", "https://api.osf.io/v2/nodes/", json.dumps({
    "data": {
        "type": "nodes",
        "attributes": {
            "title": TITLE,
            "description": DESC[:300],
            "category": "project",
            "tags": ["symbolic-regression", "astrophysics"]
        }
    }
}).encode())
comp_id = comp["data"]["id"]
print(f"  OSF: https://osf.io/{comp_id}/")

# Wiki with DOI
osf_request("POST", f"https://api.osf.io/v2/nodes/{comp_id}/wikis/", json.dumps({
    "data": {
        "type": "wikis",
        "attributes": {
            "name": "home",
            "content": f"## Links\n- Zenodo: https://doi.org/{doi}\n- GitHub: {GITHUB_REPO}"
        }
    }
}).encode())
print("  Wiki added")

print(f"\n{'='*60}")
print(f"DONE: https://doi.org/{doi}")