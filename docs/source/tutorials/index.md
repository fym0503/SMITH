# Notebook Tutorials

The tutorials are executable Jupyter notebooks for the Results sections whose real completed outputs are available in this public package. Each notebook validates pinned checksums, analyzes the result table directly, displays tables and figures, and states which parts of the manuscript require additional data or compute.

The repository keeps editable source notebooks and executed documentation copies under `docs/source/tutorials/notebooks/`. To run them locally:

```bash
python -m pip install -e '.[notebooks]'
jupyter lab docs/source/tutorials/notebooks
```

The executed notebooks shown on this site are regenerated from the source notebooks by `scripts/build_tutorial_notebooks.py`. The fixtures are copied from completed paper-workspace outputs, not synthetic substitutes. WMB source/data availability is documented separately.

```{toctree}
:maxdepth: 1

01_wmb
notebooks/regulatory_section/02_SMITH_Regulatory_Activity_executed
notebooks/ribomap_section/03_SMITH_RIBOMap_Transfer_executed
notebooks/disease_section/04_SMITH_InHouse_Disease_Transfer_executed
notebooks/agent_section/05_SMITH_Agent_Evaluation_executed
```
