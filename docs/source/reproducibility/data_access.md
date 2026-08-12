# Data and Access

Raw datasets are intentionally excluded from the Python distribution. Their expected package paths and original workspace locations are tracked in `manifests/external_data_manifest.tsv` and in the Agent dataset registry.

## Public external data

The whole-mouse-brain, C. elegans regulatory-activity, RIBOMap and STARmap analyses should ultimately use DOI- or repository-versioned downloads. A release should publish checksums and preparation commands instead of relying on `/workspace/...` paths.

## Controlled-access data

The human neurodegeneration atlas example distributes only de-identified aggregate statistics. Authorized users can place prepared H5AD files at the paths declared by the dataset registry and run the full workflow in their approved environment.

## Probe backend assets

Full feasibility screening can require transcript FASTA files, BLAST databases, genome indexes and separate tool environments. These assets belong in a download/cache layer and must not be embedded in the wheel.

## Release requirement

Before publication, every public dataset entry should include an accession or DOI, license, preparation script, expected SHA-256 checksum and the manuscript cases that consume it.
