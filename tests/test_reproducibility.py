from __future__ import annotations

import json
from pathlib import Path

import pytest

from smith.reproducibility import check_case, load_cases, run_case


EXPECTED_CASES = {
    "01_wmb",
    "02_regulatory_activity",
    "03_ribomap_transfer",
    "04_inhouse_disease",
    "05_agent",
}


def test_all_results_sections_have_ready_manifests():
    cases = load_cases()
    assert set(cases) == EXPECTED_CASES
    assert [case.order for case in cases.values()] == [1, 2, 3, 4, 5]
    for case in cases.values():
        assert "level" not in case.to_dict()
        assert check_case(case)["ready"]
        assert case.manuscript_section
        assert case.claim
        assert case.full_workflow.get("availability")


@pytest.mark.parametrize("case_id", sorted(EXPECTED_CASES - {"01_wmb"}))
def test_aggregate_reproducibility_case(case_id: str, tmp_path: Path):
    case = load_cases()[case_id]
    result = run_case(case, tmp_path / case_id)
    summary_path = Path(result["summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["case"] == case_id
    assert "reproducibility_level" not in payload
    assert payload["manuscript_section"] == case.manuscript_section
    assert payload["claim"] == case.claim


def test_wmb_reproducibility_example(tmp_path: Path):
    case = load_cases()["01_wmb"]
    result = run_case(case, tmp_path / case.id)
    assert result["selected_panel_size"] == 8
    assert result["ranking_size"] >= result["selected_panel_size"]
    assert len(result["selected_targets"]) == 8
