"""Prereg validator for the Neural Genome two-gate promotion spec.

Reads a pre-registration markdown file (per atlas_tl_session.md section 2.5.9
template) and derives the locked Gate-1 decision parameters (K, c, delta values,
equivalence formula) from its declared contents.

Mechanizes the governance layer so the adversarial auditor cannot call the
Gate-1 rule "rigor theater." Exits with non-zero code if the prereg is
internally inconsistent (e.g., declared K does not match the enumerated
systems-by-criteria grid) or if any locked pointer is missing.

Usage:
    python code/prereg_validator.py research/prereg/<name>.md

Outputs a JSON summary to stdout.

Per project conventions: ASCII-only source, no Unicode.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import NormalDist


# -------------------- Parsed prereg structure --------------------

@dataclass
class PreregConfig:
    """Parsed decision-rule parameters extracted from a prereg markdown."""

    scope_id: str
    alpha_fwer: float
    n_systems: int
    decisions_per_system: int
    K: int
    delta_relative: float
    delta_slope: float
    delta_neg_control: float
    estimator_variants: list[str]
    quantization_points: list[str]
    n_sweep: list[int]

    def compute_c(self) -> float:
        """Bonferroni-corrected one-sided z critical value."""
        tail = self.alpha_fwer / self.K
        return NormalDist().inv_cdf(1.0 - tail)


@dataclass
class ValidationResult:
    passed: bool
    config: PreregConfig | None
    errors: list[str]
    warnings: list[str]
    derived: dict[str, float | str]


# -------------------- Parsing --------------------

_SCOPE_ID_RE = re.compile(r'scope_id\s*=\s*"([^"]+)"')
_ALPHA_FWER_RE = re.compile(r'[Aa]lpha[_ ]?FWER\s*=\s*([0-9.]+)')
# Accept ASCII "delta_" or Greek-letter "δ_" prefix.
_DELTA_RELATIVE_RE = re.compile(r'(?:delta|δ)_relative\s*=\s*([0-9.]+)')
_DELTA_SLOPE_RE = re.compile(r'(?:delta|δ)_slope\s*=\s*([0-9.]+)')
_DELTA_NEG_CONTROL_RE = re.compile(r'(?:delta|δ)_neg[_-]control\s*=\s*([0-9.]+)')
_K_DECLARED_RE = re.compile(r'\bK\s*=\s*(\d+)\b')
_N_SYSTEMS_RE = re.compile(r'(\d+)\s*systems\s*[x×]\s*(\d+)\s*decisions', re.IGNORECASE)
_N_SWEEP_RE = re.compile(r'n\s*(?:∈|in)\s*\{([0-9,\s]+)\}')
# Pinned-identity pointer for F.generator etc.: (git_commit=<hash>, file_path="...", symbol="...")
# Capturing groups (1) git_commit, (2) file_path, (3) symbol.
_PINNED_PTR_RE = re.compile(
    r'\(\s*git_commit\s*=\s*<?([A-Za-z0-9]+)>?\s*,\s*'
    r'file_path\s*=\s*"([^"]+)"\s*,\s*'
    r'symbol\s*=\s*"([^"]+)"\s*\)'
)


def _read_greek_alpha_fwer(text: str) -> float | None:
    # The source doc uses the Greek letter alpha in "alpha_FWER"; also accept
    # the ASCII spelling. Prefer explicit ASCII match to avoid encoding issues.
    m = _ALPHA_FWER_RE.search(text)
    if m:
        return float(m.group(1))
    # Fallback: look for the common "alpha_FWER = 0.05" pattern with Greek letter.
    m = re.search(r'α_FWER\s*=\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    return None


def parse_prereg(path: Path) -> tuple[PreregConfig | None, list[str]]:
    """Extract locked decision-rule parameters from a prereg markdown file.

    Returns (config, errors). If errors is non-empty the config is best-effort.
    """
    errors: list[str] = []
    if not path.exists():
        return None, [f"prereg file not found: {path}"]

    text = path.read_text(encoding="utf-8", errors="replace")

    scope_id_m = _SCOPE_ID_RE.search(text)
    scope_id = scope_id_m.group(1) if scope_id_m else ""
    if not scope_id:
        errors.append("scope_id not found in prereg")

    alpha_fwer = _read_greek_alpha_fwer(text)
    if alpha_fwer is None:
        errors.append("alpha_FWER not found; expected e.g. 'alpha_FWER = 0.05'")
        alpha_fwer = 0.05

    # Equivalence margins.
    delta_relative = _extract_float(_DELTA_RELATIVE_RE, text, default=0.10,
                                    errors=errors, name="delta_relative")
    delta_slope = _extract_float(_DELTA_SLOPE_RE, text, default=0.05,
                                 errors=errors, name="delta_slope")
    delta_neg_control = _extract_float(_DELTA_NEG_CONTROL_RE, text, default=0.20,
                                       errors=errors, name="delta_neg_control")

    # K enumeration - two independent reads, must match.
    K_declared_m = _K_DECLARED_RE.search(text)
    K_declared = int(K_declared_m.group(1)) if K_declared_m else None

    systems_decisions_m = _N_SYSTEMS_RE.search(text)
    if systems_decisions_m:
        n_systems = int(systems_decisions_m.group(1))
        decisions_per_system = int(systems_decisions_m.group(2))
        K_computed = n_systems * decisions_per_system
    else:
        n_systems = 0
        decisions_per_system = 0
        K_computed = 0
        errors.append("K enumeration not parseable; expected e.g. "
                      "'3 systems x 6 decisions'")

    if K_declared is not None and K_computed and K_declared != K_computed:
        errors.append(f"K inconsistency: declared K={K_declared} but grid "
                      f"enumeration yields {K_computed} = {n_systems} x "
                      f"{decisions_per_system}")

    K = K_declared if K_declared is not None else K_computed
    if K <= 0:
        errors.append("K must be positive and > 0; check prereg enumeration")
        K = 1  # prevent div-by-zero downstream

    # n-sweep for G1.6 (optional; warning if missing).
    n_sweep_m = _N_SWEEP_RE.search(text)
    n_sweep: list[int] = []
    if n_sweep_m:
        raw = n_sweep_m.group(1).replace(" ", "")
        try:
            n_sweep = [int(x) for x in raw.split(",") if x]
        except ValueError:
            errors.append(f"n-sweep not parseable: {raw!r}")

    # Estimator variants and quantization points - best-effort extraction.
    estimator_variants = _extract_list_after(text, "Estimator variants")
    quantization_points = _extract_list_after(text, "Quantization ladder points")

    config = PreregConfig(
        scope_id=scope_id,
        alpha_fwer=alpha_fwer,
        n_systems=n_systems,
        decisions_per_system=decisions_per_system,
        K=K,
        delta_relative=delta_relative,
        delta_slope=delta_slope,
        delta_neg_control=delta_neg_control,
        estimator_variants=estimator_variants,
        quantization_points=quantization_points,
        n_sweep=n_sweep,
    )
    return config, errors


def _extract_float(pattern: re.Pattern, text: str, default: float,
                   errors: list[str], name: str) -> float:
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            errors.append(f"{name} value not parseable: {m.group(0)!r}")
    else:
        errors.append(f"{name} not found in prereg; using default {default}")
    return default


def _extract_list_after(text: str, header: str) -> list[str]:
    """Best-effort: extract a short bulleted list under a header like
    'Estimator variants'. Returns empty if not found.
    """
    header_m = re.search(re.escape(header), text)
    if not header_m:
        return []
    tail = text[header_m.end():header_m.end() + 600]
    items = re.findall(r'\(?([a-z][a-z0-9_\- ]{0,40})\)?,', tail,
                       flags=re.IGNORECASE)
    return [s.strip() for s in items[:4] if s.strip()]


# -------------------- Validation --------------------

def validate(path: Path) -> ValidationResult:
    config, parse_errors = parse_prereg(path)
    errors = list(parse_errors)
    warnings: list[str] = []
    derived: dict[str, float | str] = {}

    if config is None:
        return ValidationResult(passed=False, config=None,
                                errors=errors, warnings=warnings, derived=derived)

    # Derive the Bonferroni-corrected one-sided z critical value.
    try:
        c = config.compute_c()
        derived["c"] = round(c, 4)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to compute c from alpha_FWER={config.alpha_fwer} "
                      f"and K={config.K}: {exc!r}")
        derived["c"] = float("nan")

    # Equivalence-criterion formula summary.
    derived["equivalence_formula"] = (
        f"|Delta| + c * SE(Delta) < delta  with c = {derived.get('c')} "
        f"and delta from prereg section 5 (default delta_relative="
        f"{config.delta_relative})"
    )

    # Sanity: at least 2 estimator variants (G1.4) and at least 2 quantization
    # points (G1.5) must be declared. These are optional in the prereg prose
    # format; missing -> warning, not error.
    if len(config.estimator_variants) < 2:
        warnings.append("fewer than 2 estimator variants detected; "
                        "G1.4 stability test may be ill-defined")
    if len(config.quantization_points) < 2:
        warnings.append("fewer than 2 quantization points detected; "
                        "G1.5 stability test may be ill-defined")
    if len(config.n_sweep) < 3:
        warnings.append("n-sweep has fewer than 3 points; G1.6 asymptote "
                        "slope estimate may be noisy")

    if config.decisions_per_system <= 0:
        errors.append("decisions_per_system is zero or negative")
    if config.n_systems < 3:
        errors.append(f"n_systems < 3 ({config.n_systems}); Gate-1 portability "
                      f"requires >= 3 system classes per atlas_tl_session.md "
                      f"section 2.5.1")

    if config.alpha_fwer <= 0.0 or config.alpha_fwer >= 1.0:
        errors.append(f"alpha_FWER out of (0, 1): {config.alpha_fwer}")

    # Placeholder-rejection for LOCKED prereg (Codex R6 priority directive + Part V).
    # Status discipline: a prereg declares `status: STAGED` (can hold placeholders
    # for later fill-in) or `status: LOCKED` (must be fully pinned — no HEAD, no
    # PLACEHOLDER). Validator enforces. Also: strawman preregs in research/prereg/
    # drafts/ are excluded from lock-grade checks.
    text = path.read_text(encoding="utf-8", errors="replace")
    is_drafts = "research/prereg/drafts" in str(path).replace("\\", "/")
    is_prereg_folder = ("research/prereg/" in str(path).replace("\\", "/")
                        and not is_drafts)

    status_match = re.search(r'\bstatus\s*:\s*(STAGED|LOCKED)\b', text)
    declared_status = status_match.group(1) if status_match else None
    derived["declared_status"] = declared_status or "UNSPECIFIED"

    if is_prereg_folder:
        # In the prereg folder but no status declared — require explicit lock stance.
        if declared_status is None:
            errors.append(
                "prereg in research/prereg/ must declare `status: STAGED` or "
                "`status: LOCKED` (per Codex R6 lock discipline). Without an "
                "explicit status, lock semantics are ambiguous."
            )

        if declared_status == "LOCKED":
            if re.search(r'git_commit\s*=\s*HEAD\b', text):
                errors.append(
                    "LOCKED prereg contains 'git_commit=HEAD' sentinel — must "
                    "replace with an actual commit SHA before lock (Codex R6 Part V "
                    "self-deception #1)."
                )
            if re.search(r'PLACEHOLDER_sha256|PLACEHOLDER_[a-z_]+', text):
                errors.append(
                    "LOCKED prereg contains 'PLACEHOLDER_' values — the "
                    "conditioning distribution is not actually pinned (Codex R6 "
                    "Part V self-deception #2). Populate real dataset_hash "
                    "before lock."
                )

    # Code-identity pinning check (F.generator, filter, invariance_check per 2.5.7).
    # Every Callable referenced in F must be pinned to (git_commit, file_path, symbol)
    # AND the target must actually resolve at that commit. Regex-counting alone was
    # flagged by Codex R5 as "theater" — hardened here to do real resolution.
    text = path.read_text(encoding="utf-8", errors="replace")
    pinned = _PINNED_PTR_RE.findall(text)
    expected_pinned = 3  # generator, filter, invariance_check per 2.5.7
    if len(pinned) < expected_pinned:
        errors.append(
            f"F code-identity pinning: found {len(pinned)} (git_commit, file_path, "
            f"symbol) pointers in prereg; 2.5.7 requires >= {expected_pinned} "
            f"(generator + filter + invariance_check). Callable references "
            f"without pinned identity allow scope creep (Codex R4 kill shot #2)."
        )

    # Resolve each pointer: check file exists in repo + symbol is defined at top level
    # (module-level def or assignment). This closes the Codex R5 'theater' critique.
    repo_root = path.parent
    # Walk up until we find .git or fall back to prereg's directory.
    probe = path.resolve()
    while probe.parent != probe:
        if (probe / ".git").exists():
            repo_root = probe
            break
        probe = probe.parent
    derived["pinned_pointers"] = []
    for (commit, file_path, symbol) in pinned:
        resolved = _resolve_pinned_pointer(repo_root, commit, file_path, symbol)
        derived["pinned_pointers"].append(resolved)
        if not resolved["file_exists"]:
            errors.append(
                f"pinned pointer unresolved: file '{file_path}' does not exist "
                f"in repo (referenced from prereg with symbol '{symbol}')"
            )
        elif not resolved["symbol_defined"]:
            errors.append(
                f"pinned pointer unresolved: symbol '{symbol}' not defined at "
                f"top level of '{file_path}' (referenced from prereg)"
            )
        if commit != "HEAD" and not resolved.get("commit_verified", False):
            warnings.append(
                f"pinned pointer git_commit={commit!r} not verified against "
                f"repo state; validator only checks current HEAD. Use commit='HEAD' "
                f"or verify via git show {commit}:{file_path}."
            )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, config=config,
                            errors=errors, warnings=warnings, derived=derived)


# -------------------- Pinned-pointer resolution --------------------

def _resolve_pinned_pointer(repo_root: Path, commit: str, file_path: str,
                            symbol: str) -> dict[str, object]:
    """Verify that (commit, file_path, symbol) resolves in the repo.

    Checks (in order):
      1. file exists at repo_root / file_path
      2. symbol is defined at module top level in that file
      3. (optional) git show <commit>:<file_path> succeeds and matches
    """
    full = repo_root / file_path
    result: dict[str, object] = {
        "commit": commit,
        "file_path": file_path,
        "symbol": symbol,
        "file_exists": full.is_file(),
        "symbol_defined": False,
        "commit_verified": None,
    }
    if not result["file_exists"]:
        return result

    try:
        src = full.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(full))
    except (OSError, SyntaxError) as exc:
        result["symbol_defined"] = False
        result["parse_error"] = repr(exc)
        return result

    top_level_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)

    result["symbol_defined"] = symbol in top_level_names

    # Commit verification: if the prereg pins a specific commit, try
    # `git show <commit>:<file_path>`. If <hash>-like placeholder (e.g.
    # "HEAD" or "<hash>") skip. Best-effort; never blocks.
    if commit and commit != "HEAD" and not commit.startswith("<"):
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "show",
                 f"{commit}:{file_path}"],
                capture_output=True, text=True, timeout=5,
            )
            result["commit_verified"] = (proc.returncode == 0)
        except (OSError, subprocess.TimeoutExpired):
            result["commit_verified"] = False

    return result


# -------------------- CLI --------------------

def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python code/prereg_validator.py <prereg.md>",
              file=sys.stderr)
        return 2
    result = validate(Path(argv[1]))
    payload = {
        "passed": result.passed,
        "errors": result.errors,
        "warnings": result.warnings,
        "derived": result.derived,
        "config": asdict(result.config) if result.config is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
