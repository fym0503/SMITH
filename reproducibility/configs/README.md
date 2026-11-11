# Tutorial profiles and training settings

`tutorial_profiles.yaml` defines the three task profiles shown to users: `TF_profile`, `ribomap_profile` and `Agent_profile`. Set `SMITH_TUTORIAL_PROFILE` to select a profile when supported by the notebook or CLI.

The task profile determines the model implementation and recorded parameters. The implementation registry in `training_models.yaml` is an execution detail; notebooks and Agent model listings use the task-profile names.

## RIBOMap

`ribomap_paper.json` contains four recorded SMITH configurations. The notebook trains each configuration once and evaluates the selected panels with uniform 5-nearest-neighbour classification over five target splits. Its comparison table places the computed means beside the recorded results.

## Liver Agent

`Agent_profile` uses five Visium references, five training seeds, 200 epochs, batch size 64 and fixed 0.5/0.5 source/reference weights. The source label is `Cell_Type_final`; the reference label is `cell_type`. MERFISH is used for evaluation.

## C. elegans TF

`tf_paper.json` configures five TF lineage splits. Each split trains one ranking with 90 epochs, seed 8013 and the full training data, then evaluates panels of 32, 64 and 128 TFs. `tf_transfer_paper.json` configures scRNA-seq-based panel transfer.
