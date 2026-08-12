# Provenance and Artifacts

Each case is defined by a YAML file under `reproducibility/manifests/`. The manifest records:

- manuscript section and figure;
- representative claim and example scope;
- expected runtime;
- pinned inputs and declared outputs;
- full-paper source workflows and data-access limitations.

The bundled fixtures under `reproducibility/fixtures/` are compact derived tables copied from completed manuscript runs. They should remain immutable for a tagged paper release. A changed fixture requires a new checksum, a provenance note and review of any affected manuscript claim.

Generated artifacts belong under `outputs/reproducibility/` and are not packaged. CI should run `smith-repro check` and all five examples.

The paper release environment is pinned in `requirements-reproducibility.txt`. The general package dependencies in `pyproject.toml` remain intentionally less restrictive for library users.
