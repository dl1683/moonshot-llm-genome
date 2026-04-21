"""Assemble the workshop-paper fragments in `research/paper_*_draft.md` into
a single reviewable `research/PAPER.md` with inline figure references.

Per CLAUDE.md §3.3 (no new file unless it creates a reusable boundary):
this script formalises the paper-build step so any future draft updates
re-assemble the canonical PAPER.md deterministically. The alternative —
hand-editing a concatenated file — would drift between the fragments
and the assembly.

Stripping rules:
  - Drop the "**Status:** DRAFT ..." header line from each fragment.
  - Drop any section with header equal to "Integration notes", "Known gaps",
    "Word-count self-check", "Draft author list", "Open-source release plan",
    "Blockers for lock". These are process-meta, not paper content.
  - Keep everything else.

Figure injection points:
  - Fig 1 (C(k) cross-architecture) — after §4.1 paragraph, before §4.2
  - Fig 2 (G2.4 causal ablation) — at end of §4.4
  - Fig 3 (Geometry → Efficiency) — at end of §5.5 (Table 8)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FRAG = _ROOT / "research"
_OUT = _ROOT / "research" / "PAPER.md"
_FIG_REL = "../results/figures"  # relative from research/PAPER.md

_SKIP_H2 = {
    "Integration notes", "Known gaps flagged for paper reviewers",
    "Word-count self-check", "Draft author list",
    "Open-source release plan", "Blockers for lock",
    "Known gaps",
}


def strip_meta(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    skip_level = 0
    for line in lines:
        stripped = line.strip()
        # Drop H1 titles that are fragment-meta ("# §X Foo — Draft Prose ...")
        if line.startswith("# ") and "Draft Prose" in line:
            continue
        if stripped.startswith("**Status:") and "DRAFT" in stripped:
            continue
        if line.startswith("## "):
            heading = line[3:].strip().rstrip(":")
            name_only = re.sub(r"^§\S+\s+", "", heading)
            if name_only in _SKIP_H2 or heading in _SKIP_H2:
                skip_level = 2
                continue
            else:
                skip_level = 0
        elif skip_level == 2:
            continue
        out.append(line)
    txt = "\n".join(out)
    # Drop "---" lines followed by "**Word-count self-check**" blocks through EOF or next H2
    txt = re.sub(r"\n+---\n+\*\*Word-count self-check.*?(?=\n## |\Z)",
                 "\n", txt, flags=re.DOTALL)
    # Drop standalone "**Word-count self-check ... ✓**" lines
    txt = re.sub(r"^\*\*Word-count self-check.*?$", "", txt,
                 flags=re.MULTILINE)
    # Drop leading "---" that came from fragment separators
    txt = re.sub(r"^\s*---\s*\n", "", txt)
    # Collapse 3+ consecutive blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def fragment(name: str) -> str:
    p = _FRAG / f"paper_{name}_draft.md"
    if not p.exists():
        raise FileNotFoundError(p)
    return strip_meta(p.read_text(encoding="utf-8"))


def build():
    # Title + abstract (from PAPER_OUTLINE.md)
    outline = (_FRAG / "PAPER_OUTLINE.md").read_text(encoding="utf-8")
    # Extract abstract paragraph
    m = re.search(r"## Abstract.*?\n\n(.*?)\n\n---", outline, re.DOTALL)
    abstract = m.group(1).strip() if m else "(abstract pending)"

    title = "Geometry, Not Scale: Cross-Architecture Portability and Compression-Gating of a Local-Neighborhood Invariant in Trained Neural Networks"

    methods = fragment("methods")
    results = fragment("results")
    discussion = fragment("discussion")
    intro_rw = fragment("intro_relwork")
    conclusion = fragment("conclusion")

    # Inject figures at known anchors
    # Figure 1 goes right before §4.2 (just after the table of §4.1)
    results = re.sub(
        r"(## 4\.2 Not a random-geometry artifact)",
        r"""![Figure 1](""" + _FIG_REL + r"""/genome_fig1_ck_cross_architecture.png)

**Figure 1.** `C(X, k)` curves across five trained architectures at mid-depth, error bars = std across 3 stimulus-resample seeds. Curves are nearly homothetic across the full `k ∈ [3, 130]` range; monotonically increasing in `k` on every system (falsifying the locked v1 derivation; see §4.5).

\1""",
        results, count=1,
    )
    # Figure 2 at end of §4.4 (before §4.5)
    results = re.sub(
        r"(## 4\.5 Functional-form identification)",
        r"""![Figure 2](""" + _FIG_REL + r"""/genome_fig2_causal_ablation.png)

**Figure 2.** Gate-2 G2.4 causal-ablation effect on three text architectures. topk (red) substantially exceeds random-10d (grey) and top-PC-10 (blue) at every `λ`; the 5% pre-registered `δ_causal` threshold is marked. topk is monotone in `λ` on every (system, depth) cell; specificity 34–66×.

\1""",
        results, count=1,
    )
    # Figure 3 at end of §5.5 — append at the very end of discussion
    discussion = discussion + "\n\n![Figure 3](" + _FIG_REL + "/genome_fig3_geometry_efficiency.png)\n\n**Figure 3.** Geometry → Efficiency probe on Qwen3-0.6B. R² of the power-law fit (green, left axis) decreases monotonically as weight quantization tightens (FP16 → Q8 → Q4); relative NLL increase (red, right axis) tracks the same direction. R² is the clean geometric early-warning signal for compression-induced capability loss.\n"

    body = f"""# {title}

**Authors:** Dev (CMC / AI Moonshots)

**Version:** 2026-04-21, preprint draft.

**Repository:** `github.com/dl1683/moonshot-llm-genome` (scheduled for open-source release at submission).

## Abstract

{abstract}

---

{intro_rw}

---

{methods}

---

{results}

---

{discussion}

---

{conclusion}

---

## Reproducibility

All results in this paper are backed by a commit-pinned pre-registration (`research/prereg/genome_knn_k10_portability_2026-04-21.md` LOCKED at `62338b8`; Gate-2 G2.4 prereg LOCKED at `03da4d5`; Batch-2 prereg LOCKED at `3e8d395`) and a machine-executable validator (`code/prereg_validator.py`). The full atlas is reproducible from:

- **Code:** `code/genome_*.py`
- **Data hashes:** text `6c6ccf844f9ec8b6…9316f7` (C4-clean n=2000 × 3 seeds); vision `0a3af317f9775044…6bb02f` (ImageNet-val n=2000 × 3 seeds)
- **Hardware envelope:** `COMPUTE.md` (≤22 GB VRAM, ≤56 GB RAM, ≤4 h per experiment, single RTX 5090 laptop)
- **Ledger:** `experiments/ledger.jsonl` — every numeric claim in this paper maps to a ledger entry.
- **Claim → evidence map:** `research/CLAIM_EVIDENCE_MAP.md`
- **Locked derivation artefact (falsified):** `research/derivations/knn_clustering_universality.md`

## Acknowledgements

An earlier version of this work was reviewed by our own senior architectural agent (OpenAI Codex, two fresh sessions at milestone + strategic-adversarial gates). Their criticisms materially improved the paper — especially the scope-metadata bug they flagged for vision atlas rows (fix at commit `f4973dc`), the SE calibration analysis (fix at commit `9bbee73`), and the Geometry → Efficiency probe scoping (strategic verdict at commit `f625ca9`).
"""
    _OUT.write_text(body, encoding="utf-8")
    print(f"wrote {_OUT}  ({len(body.split())} words, {_OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
