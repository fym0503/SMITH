# Run provenance

Every runnable workflow writes `run_manifest.json` with CLI configuration, resolved input paths, input sizes and SHA-256 values, training artifacts, generated panels and evaluation outputs. Outputs are always rooted at `--output-dir`; workflows do not depend on `/workspace/fanyimin/...` paths.

The versioned data manifest records the intended archive contents independently from any one server. Optional aggregate tables in `reproducibility/reference_outputs/` preserve historical manuscript summaries for comparison, but are not accepted as tutorial inputs.
