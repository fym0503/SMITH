"""Reader-facing introductions and explanations shared by tutorial builds."""

COPY = {
    'tf': {
        0: '# Regulatory programs and cross-modality transfer in C. elegans\n\nTrain TF panels across five lineage splits. Evaluate cell identity and developmental timing, compare TF/scRNA correlation structure, and test scRNA-seq-to-TF transfer.\n\nThis notebook uses `TF_profile`: 90 epochs per split, seed 8013 and the full TF training data.',
        1: '## Biological question\n\nWhich TFs preserve cell identity and developmental timing in unseen cell lineages? Compare panels selected from TF activity with panels selected from a scRNA-seq reference.',
        2: '## Prepare the input data\n\nPlace the five TF train/test splits, module annotation table and scRNA-seq reference listed in `reproducibility/data_manifest.yaml` under one data directory. Set `SMITH_TUTORIAL_DATA` to that directory before starting Jupyter. The manifest records expected file sizes and SHA-256 checksums.',
        5: '## Load the activity atlas and annotations\n\nLoad the five lineage-separated train/test pairs, the spatiotemporal module annotation and the scRNA-seq reference. TF activity data include cell-type labels and developmental time.',
        7: '## Train SMITH and select TF panels\n\nTrain one model per lineage split using reconstruction, classification and time objectives. Extract panels of 32, 64 and 128 targets from each ranking.',
        9: '## Evaluate cell identity and developmental time\n\nApply each selected panel to the matching held-out lineage split. Save cell-type accuracy and developmental-time correlation for each panel size.',
        11: '### Cell identity and developmental-time metrics\n\nCompare cell-type accuracy and developmental-time correlation across panel sizes.',
        12: '## Spatiotemporal TF modules\n\nThe atlas annotation groups regulators by tissue lineage and developmental phase. The heatmap summarizes these module annotations.',
        14: '## TF correlation structure\n\nAggregate shared TFs over matched lineages in scRNA-seq and TF activity. Compare their TF–TF correlation matrices.',
        16: '## Transfer from scRNA-seq to TF activity\n\nTrain a panel ranking from scRNA-seq using its recorded configuration. Evaluate its panels on each held-out TF split and compare with panels selected directly from TF activity.',
        18: '## Plot the results\n\nPlot prediction accuracy, developmental-time correlation, module structure, TF correlation and cross-modality transfer.',
        20: '## Save the run details\n\nSave input checksums, training parameters and output paths in `run_manifest.json`.',
        22: '## Training configurations\n\nThe configurations in `reproducibility/configs/tf_paper.json` and `reproducibility/configs/tf_transfer_paper.json` define the TF and scRNA-seq training runs.',
    },
    'ribomap': {
        0: '# Cross-modality brain panel transfer to RIBOMap\n\nUse `ribomap_profile` to train four SMITH configurations on Deep-RIBOmap and STARmap references. Evaluate panels of 32, 64 and 128 genes on a RIBOMap target using five random 80/20 train–test splits and uniform 5-nearest-neighbour classification. The comparison table pairs the resulting accuracy means with the paper values.',
        1: '## Prepare the input data\n\nPlace the Deep-RIBOmap, STARmap and RIBOMap target H5AD files listed in `reproducibility/data_manifest.yaml` under one data directory. Set `SMITH_TUTORIAL_DATA` to that directory before starting Jupyter. The manifest records expected file sizes and SHA-256 checksums.',
        2: '## Load the reference and target datasets',
        5: '## Train SMITH and evaluate panel transfer',
        17: '### RIBOMap expression support\n\nCalculate expression bias for genes selected by each source modality, shared genes and background genes. The displayed distribution uses the 128-gene panels.',
        19: '## Save the run details\n\nSave the training configuration, evaluation settings and output paths in `run_manifest.json`.',
    },
    'agent': {
        0: '# Liver panel design with five spatial references\n\nUse `Agent_profile` to train source and reference SMITH models across five training seeds. Combine reference rankings with equal source/reference weights and evaluate the resulting panels in MERFISH.',
        1: '## Prepare the input data\n\nPlace the liver snRNA-seq, MERFISH and five Visium H5AD files listed under `05_agent` in `reproducibility/data_manifest.yaml` under one data directory. Set `SMITH_TUTORIAL_DATA` to that directory before starting Jupyter.',
        2: '## Load the source, spatial references and MERFISH data',
        7: '## Aggregate source and reference rankings\n\nCombine the source ranking and spatial-reference ranking with equal weights. Retain genes supported by at least two spatial references.',
        15: '### Figure 6f: probe-screen pass rates\n\nPlot the recorded ODT, OligoMiner and ProbeDealer results for the 12,160-gene candidate set.',
        17: '### Figure 6g: ProbeDealer examples\n\nCompare target-compatible probes, known off-targets and unresolved hits for ten representative genes using the recorded probe annotations.',
        19: '## Save the run details\n\nSave the training configuration, evaluation settings and output paths in `run_manifest.json`.',
    },
}


def apply_reader_copy(notebook, case):
    for index, text in COPY[case].items():
        if notebook.cells[index].cell_type != 'markdown':
            raise ValueError(f'{case}: expected markdown at cell {index}')
        notebook.cells[index].source = text
    last = notebook.cells[-1]
    if case == 'tf':
        pass
    elif case == 'agent':
        last.source = ('## Run the Figure 6f-g probe-screening workflow\n\n'
                       'Configure the human transcriptome index and the ODT, OligoMiner and ProbeDealer backends, then run:\n\n'
                       '```bash\n'
                       'python reproducibility/workflows/agent/run_tutorial.py \\\n'
                       '  --data-root data/tutorials \\\n'
                       '  --output-dir outputs/paper/agent \\\n'
                       '  --panel-sizes 32,64,128 \\\n'
                       '  --training-seeds 1,2,3,4,5 \\\n'
                       '  --run-probe-feasibility\n'
                       '```')
    return notebook
