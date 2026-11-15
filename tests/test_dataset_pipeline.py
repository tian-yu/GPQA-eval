"""CPU-only unit tests for src/coverage.py and src/build_dataset.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset  # noqa: E402
import coverage  # noqa: E402
import decontam  # noqa: E402


def test_coverage_prefers_diverse_texts():
    texts = [
        "quantum mechanics hamiltonian eigenvalue spin operator wavefunction",
        "quantum mechanics hamiltonian eigenvalue spin operator wavefunction",  # duplicate
        "organic chemistry sn2 nucleophilic substitution stereochemistry inversion",
        "molecular biology crispr cas9 endonuclease guide rna transcription",
    ]
    idx = coverage.select_indices(texts, k=3, norm="sqrt")
    assert set(idx) == {0, 2, 3}


def test_parse_mcq_and_format():
    q = "Which particle has zero rest mass?\nA) Electron\nB) Proton\nC) Photon\nD) Neutron"
    parsed = build_dataset.parse_mcq(q)
    assert parsed is not None
    stem, choices = parsed
    assert stem == "Which particle has zero rest mass?"
    assert choices == ["Electron", "Proton", "C) Photon"[3:], "Neutron"]

    sol = build_dataset.format_gpqa_solution("Photons are massless gauge bosons.\n\\boxed{C}", "C")
    assert sol.endswith("The answer is (C).")
    assert "\\boxed" not in sol


def test_question_bucket_deterministic():
    b1 = build_dataset.question_bucket("What is Newton's second law?")
    b2 = build_dataset.question_bucket("  what  is newton's   second law? ")
    assert b1 == b2
    assert 0.0 <= b1 < 1.0


def test_decontaminate_rows_drops_overlap():
    gpqa_q = "A synthetic placeholder question about the spin states of a hypothetical particle in a harmonic potential well"
    grams = decontam.ngrams(gpqa_q)
    rows = [
        {"messages": [{"role": "user", "content": "Explain photosynthesis."}, {"role": "assistant", "content": "Plants convert light."}]},
        {"messages": [{"role": "user", "content": gpqa_q}, {"role": "assistant", "content": "Answer"}]},
    ]
    clean, rep = build_dataset.decontaminate_rows(rows, grams)
    assert rep["train_rows_flagged"] == 1
    assert len(clean) == 1
