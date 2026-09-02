"""
tests/test_generator_leakage.py

Confirms the physical separation between ml/latent_process.py (the only
module allowed to know the ground-truth latent coercion intensity `C`) and
ml/generate.py's OBSERVABLE output path, per the honest-eval design
documented in both files' module docstrings.
"""
from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest

import generate
import latent_process

LATENT_ONLY_COLUMNS = {
    "C", "is_candidate", "raw_label_pre_noise", "label_flipped_by_noise",
}


# --------------------------------------------------------------------------- #
# Static check: generate.py's actual import statements, not its docstring
# --------------------------------------------------------------------------- #

def test_generate_only_imports_the_documented_functions_from_latent_process():
    """generate.py's own docstring claims it only ever imports
    generate_ground_truth, LatentProcessConfig, and diagnostics from
    latent_process -- never C itself. Verify this by parsing generate.py's
    real `from latent_process import ...` statement(s), not by trusting
    the docstring."""
    source = inspect.getsource(generate)
    tree = ast.parse(source)

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "latent_process":
            imported_names.update(alias.name for alias in node.names)

    assert imported_names == {"generate_ground_truth", "LatentProcessConfig", "diagnostics"}, (
        f"generate.py imports {imported_names!r} from latent_process -- expected "
        f"exactly {{'generate_ground_truth', 'LatentProcessConfig', 'diagnostics'}}. "
        f"Any additional import (especially of `C`) is a leakage regression."
    )


def test_C_is_not_a_name_in_generates_module_namespace():
    assert not hasattr(generate, "C"), (
        "generate module has a top-level attribute `C` -- the latent coercion "
        "intensity must never be importable/accessible from generate.py's namespace."
    )


# --------------------------------------------------------------------------- #
# Runtime check: build a small regime and inspect the actual dataframes
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_regime():
    """A reduced-size regime (a couple hundred rows) so this runs in well
    under a second while still exercising the real build_regime() pipeline
    end-to-end."""
    small_obs_cfg = replace(
        generate.REGIME_A,
        regime_name="regime_test_small",
        n_rows=250,
        n_users=60,
        n_payees=30,
    )
    small_latent_cfg = latent_process.LatentProcessConfig(candidate_rate=0.08, seed=42)
    observed, full_latent, diag = generate.build_regime(small_obs_cfg, small_latent_cfg)
    return observed, full_latent, diag


def test_observed_output_has_no_latent_derived_columns(small_regime):
    observed, _full_latent, _diag = small_regime
    leaked = LATENT_ONLY_COLUMNS & set(observed.columns)
    assert not leaked, (
        f"the OBSERVED dataframe (the only file features.py/train.py may read) "
        f"contains latent-only column(s) {leaked} -- these must only ever appear "
        f"in the *_full_latent.csv diagnostics file."
    )
    assert "_is_mule_payee" not in observed.columns, (
        "internal helper column `_is_mule_payee` leaked into the observed "
        "dataframe -- it is documented as dropped before saving."
    )


def test_full_latent_diagnostics_file_does_contain_C(small_regime):
    """Contrast check: the *_full_latent.csv diagnostics dataframe SHOULD
    have C -- that's the point of it existing separately. If this ever
    stops being true, the leakage test above would be vacuous (testing a
    dataframe that never had C to begin with)."""
    _observed, full_latent, _diag = small_regime
    assert "C" in full_latent.columns, (
        "full_latent dataframe is missing `C` -- if this file never carries C, "
        "the observed-vs-full-latent split has no diagnostic purpose and the "
        "leakage assertion above proves nothing."
    )
    assert "is_candidate" in full_latent.columns


def test_observed_and_full_latent_cover_the_same_transactions(small_regime):
    """Confirm observed/full_latent are two views of the SAME rows (same
    txn_ids), not independently-sampled datasets that would make the
    leakage comparison above meaningless."""
    observed, full_latent, _diag = small_regime
    assert set(observed["txn_id"]) == set(full_latent["txn_id"]), (
        "observed and full_latent dataframes cover different transactions -- "
        "they should be the same rows with/without latent columns."
    )


def test_id_columns_still_present_for_downstream_entity_disjoint_split(small_regime):
    """Sanity guard: user_id/payee_id/era are NOT latent-derived (they're
    generation-time bookkeeping, independent of C/label) and must remain
    in observed -- train.py's split logic depends on them."""
    observed, _full_latent, _diag = small_regime
    for col in ("user_id", "payee_id", "era", "txn_id", "label"):
        assert col in observed.columns, (
            f"expected non-latent column {col!r} to be present in the observed "
            f"dataframe, but it was missing."
        )
