# Source Manifest

This clean package was assembled from the local SMITH workspace.

| Package path | Source path | Notes |
|---|---|---|
| `src/smith/` | `/workspace/fanyimin/SMITH_tool-main/Smith/` | Core SMITH optimizer package, normalized to lowercase |
| `scripts/main.py` | `/workspace/fanyimin/SMITH_tool-main/main.py` | Original training entry point |
| `scripts/eval.py` | `/workspace/fanyimin/SMITH_tool-main/eval.py` | Original evaluation helper |
| `scripts/submit_eval.py` | `/workspace/fanyimin/SMITH_tool-main/submit_eval.py` | Original batch/evaluation helper |
| `src/smith_agent/` | `/workspace/fanyimin/smith-agent/src/smith_agent/` | Agentic workflow package |
| `configs/agent/` | `/workspace/fanyimin/smith-agent/configs/` | Agent registries and policies |
| `scripts/agent_examples/` | `/workspace/fanyimin/smith-agent/scripts/` | Agent analysis scripts |
| `docs/source/agent/` | `/workspace/fanyimin/smith-agent/docs/` | Agent documentation |
| `src/smith_agent/feasibility/` | `/workspace/fanyimin/smith_interface/smith_interface/` | Feasibility backends merged into SMITH-Agent |
| `src/smith_agent/probedealer/` | `/workspace/fanyimin/smith_interface/probedealer_py/` | ProbeDealer helpers merged into SMITH-Agent |
| `docs/source/interface_readme.md` | `/workspace/fanyimin/smith_interface/README.md` | Historical feasibility documentation |
| `docs/source/baselines_readme.md` | `/workspace/fanyimin/SMITH_baselines/README.md` | Baseline documentation |

Excluded by design:

* local conda environments
* H5AD datasets
* historical benchmark/HPO outputs
* manuscript PDF build artifacts
* Python cache files
