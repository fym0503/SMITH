# Data and access

`reproducibility/data_manifest.yaml` is the versioned source of truth for tutorial files, sizes, SHA-256 checksums, resource estimates and archive metadata. Raw H5AD data are not stored in Git or the Python wheel.

The real source files and individual-file checksums have been inventoried. The
archive checksums and Zenodo record URL remain `null` until the expanded bundles
are built, upstream licenses are checked, and an authorized user publishes them.
`CC BY-NC 4.0` is a target bundle license, not a claim that every upstream
dataset has already been relicensed.

After the Zenodo record is published, download and verify a case with:

```bash
python scripts/download_tutorial_data.py --case 03_ribomap_transfer --data-root data/tutorials
smith-repro check 03_ribomap_transfer --data-root data/tutorials
```

The downloader resumes partial files, skips verified existing data, checks archive and individual-file checksums, rejects unsafe archive paths/links, and removes incomplete extraction directories after failure.

Build a local archive before an authorized Zenodo upload with `scripts/build_tutorial_archives.py`. That command validates every source file against the manifest and prints the final archive size and checksum; it never uploads automatically.

Once the upstream licenses are confirmed and `ZENODO_TOKEN` is available in the
shell, publish the three validated archives with:

```bash
for case in 02_regulatory_activity 03_ribomap_transfer 05_agent; do
  python scripts/build_tutorial_archives.py \
    --case "$case" \
    --data-root /path/to/tutorial-data \
    --output-dir zenodo-archives
done

python scripts/publish_tutorial_data.py \
  --archive-dir zenodo-archives \
  --publish
```

The publisher creates one Zenodo dataset record, uploads one archive per public
case, and writes the record/file URLs and archive checksums back to
`reproducibility/data_manifest.yaml`. It refuses pending upstream licenses by
default and deletes an incomplete deposition if an upload fails.

Whole mouse brain remains unavailable. The controlled in-house disease chapter and inputs are not included in the public case registry or tutorials.
