# Paper Examples

The examples follow the five main Results sections rather than the internal module layout. Each example runs one representative analysis and writes a machine-readable `summary.json`.

Start by checking every pinned input:

```bash
smith-repro check
```

Run a single chapter:

```bash
smith-repro run 03_ribomap_transfer
```

Outputs are written under `outputs/reproducibility/<case>/` by default. The compact fixtures are derived tables, not raw biological data. See the reproducibility matrix before interpreting an example output as a complete figure reproduction.

| Example | Manuscript analysis |
|---|---|
| `01_wmb` | Multi-objective panel selection |
| `02_regulatory_activity` | TF and miRNA activity preservation |
| `03_ribomap_transfer` | RIBOMap and STARmap transfer |
| `04_inhouse_disease` | Human disease transfer robustness |
| `05_agent` | Multi-reference ranking and probe feasibility |
