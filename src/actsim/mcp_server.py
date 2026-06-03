"""Model Context Protocol (MCP) server for the actsim package.

This server exposes every public capability of actsim as MCP tools so that an
LLM client (Claude Desktop, Claude Code, etc.) can drive the actuarial toolkit
end to end: fit distributions, draw samples, run frequency/severity Monte Carlo
simulations (with optional copula dependence), build correlated multivariate
simulations, simulate a policy book into claims and development triangles, and
inspect or update the package configuration.

The tools are intentionally self-contained (stateless): each call accepts the
data it needs either inline (as JSON lists/records) or via a file path, runs the
underlying actsim objects, and returns JSON-serialisable results. Plot helpers
render with a non-interactive backend and save a PNG to disk, returning its path.

Run it with::

    actsim-mcp                # stdio transport (default, for Claude Desktop)
    actsim-mcp --transport sse --port 8000

or ``python -m actsim.mcp_server``.
"""
from __future__ import annotations

# Matplotlib must be switched to a non-interactive backend *before* any actsim
# module (which imports pyplot at import time) is imported, otherwise the plot
# tools would try to open a GUI window inside the server process.
import matplotlib

matplotlib.use("Agg")

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcp.server.fastmcp import FastMCP

from actsim import (
    ClaimSimulator,
    DistributionFitter,
    StochasticSimulator,
    load_config,
)
from actsim.cli import (
    ALL_FIT_METRICS,
    _coerce_param_cell,
    _load_series,
    _tvar,
)

mcp = FastMCP(
    "actsim",
    instructions=(
        "Actuarial risk modelling and simulation toolkit. Use these tools to fit "
        "probability distributions to loss data, draw samples, run aggregate "
        "frequency/severity Monte Carlo simulations (optionally with copula "
        "dependence or a correlation matrix across lines of business), simulate a "
        "policy book into individual claims and loss-development triangles, and "
        "inspect or edit the package configuration. Data can be supplied inline as "
        "JSON or by pointing at a CSV/JSON file path on the server."
    ),
)


# ---------------------------------------------------------------------------
# Serialisation / input helpers
# ---------------------------------------------------------------------------

Number = Union[int, float]


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars and containers to plain Python.

    The result is always JSON-serialisable (NaN/Inf are turned into ``None``).
    """
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return _df_records(obj)
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (int, str)):
        return obj
    return str(obj)


def _df_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-safe row dicts."""
    if df is None:
        return []
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append({str(k): _to_native(v) for k, v in row.items()})
    return records


def _series_from_input(
    data: Optional[Sequence[Number]],
    input_path: Optional[str],
    column: Optional[str],
) -> pd.Series:
    """Resolve a univariate numeric Series from inline data or a CSV/JSON file."""
    if data is not None and input_path:
        raise ValueError("Provide either `data` or `input_path`, not both.")
    if data is not None:
        s = pd.to_numeric(pd.Series(list(data)), errors="coerce").dropna()
        if s.empty:
            raise ValueError("`data` contained no numeric values.")
        return s.reset_index(drop=True)
    if input_path:
        return _load_series(Path(input_path), column).reset_index(drop=True)
    raise ValueError("Supply input via `data` (a list of numbers) or `input_path`.")


def _params_tuple(params: Sequence[Number]) -> tuple:
    """Coerce a JSON list of distribution parameters into a numeric tuple."""
    out: List[Number] = []
    for p in params:
        v = float(p)
        out.append(int(v) if v.is_integer() else v)
    return tuple(out)


def _fit_series(
    series: pd.Series,
    distributions: Optional[List[str]],
    metrics: Optional[List[str]],
    truncate: Optional[Dict[str, Any]] = None,
) -> DistributionFitter:
    """Build, optionally truncate, and fit a DistributionFitter."""
    fitter = DistributionFitter(
        data=series,
        distributions=distributions,
        metrics=metrics or ALL_FIT_METRICS,
    )
    if truncate:
        fitter.truncate_data(
            remove_values=truncate.get("remove_values"),
            lower=truncate.get("lower"),
            upper=truncate.get("upper"),
            q_low=truncate.get("q_low"),
            q_high=truncate.get("q_high"),
        )
    fitter.fit()
    return fitter


AVAILABLE_DISTRIBUTIONS = list(DistributionFitter([0.0]).available_distributions.keys())
COPULA_TYPES = ["gaussian", "frank", "gumbel", "clayton"]


# ---------------------------------------------------------------------------
# Discovery / metadata tools
# ---------------------------------------------------------------------------

@mcp.tool()
def version() -> Dict[str, Any]:
    """Return the installed actsim version and a catalogue of capabilities."""
    from importlib import metadata

    try:
        ver = metadata.version("actsim")
    except metadata.PackageNotFoundError:
        ver = "unknown"
    return {
        "package": "actsim",
        "version": ver,
        "available_distributions": AVAILABLE_DISTRIBUTIONS,
        "fit_metrics": ALL_FIT_METRICS,
        "copula_types": COPULA_TYPES,
    }


@mcp.tool()
def list_distributions() -> Dict[str, Any]:
    """List the distributions, fit metrics and copulas supported by actsim."""
    return {
        "distributions": AVAILABLE_DISTRIBUTIONS,
        "fit_metrics": ALL_FIT_METRICS,
        "copula_types": COPULA_TYPES,
        "notes": (
            "Frequency distributions are typically 'poisson' or 'negative binomial'; "
            "the rest are severity distributions. Parameters follow the actstats "
            "convention (e.g. lognormal=(mu, sigma), normal=(loc, scale))."
        ),
    }


# ---------------------------------------------------------------------------
# Configuration tools (load_config / Config)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the actsim configuration (default distributions and metrics).

    Args:
        config_path: Optional path to a custom YAML config. Defaults to the
            packaged config.yaml.

    Returns:
        The configuration contents and the resolved file path on disk.
    """
    cfg = load_config(config_path)
    return {"file_path": cfg._file_path, "config": _to_native(cfg._data)}


@mcp.tool()
def update_config(updates: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """Update keys in a YAML config file and persist them to disk.

    Args:
        updates: Mapping of config keys to new values (merged into existing data).
        config_path: Optional path to the YAML config to modify. Defaults to the
            packaged config.yaml.

    Returns:
        The updated configuration contents.
    """
    cfg = load_config(config_path)
    cfg.update(updates)
    return {"file_path": cfg._file_path, "config": _to_native(cfg._data)}


# ---------------------------------------------------------------------------
# DistributionFitter tools
# ---------------------------------------------------------------------------

@mcp.tool()
def fit_distributions(
    data: Optional[List[float]] = None,
    input_path: Optional[str] = None,
    column: Optional[str] = None,
    distributions: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    select_metric: str = "aic",
    truncate_remove: Optional[List[float]] = None,
    truncate_lower: Optional[float] = None,
    truncate_upper: Optional[float] = None,
    truncate_q_low: Optional[float] = None,
    truncate_q_high: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit candidate probability distributions to univariate data.

    Computes, for every candidate distribution, the fitted parameters plus
    goodness-of-fit metrics (aic, bic, log_likelihood, chisquare, ks) and reports
    the best fit under each metric.

    Args:
        data: Inline list of numeric observations. Mutually exclusive with input_path.
        input_path: Path to a CSV/JSON file containing the data.
        column: Column to use when the file has more than one column.
        distributions: Candidate distribution names (defaults to all available).
        metrics: Metrics to compute / drive selection (defaults to all).
        select_metric: Metric used to report the single best fit (default 'aic').
        truncate_remove: Values to drop before fitting (e.g. [0, -999]).
        truncate_lower / truncate_upper: Keep observations within [lower, upper].
        truncate_q_low / truncate_q_high: Quantile cutoffs (e.g. 0.01 / 0.99).

    Returns:
        n_observations, the per-distribution fit summary, best fit per metric,
        and the best fit under select_metric.
    """
    series = _series_from_input(data, input_path, column)
    truncate = {
        "remove_values": list(truncate_remove) if truncate_remove else None,
        "lower": truncate_lower,
        "upper": truncate_upper,
        "q_low": truncate_q_low,
        "q_high": truncate_q_high,
    }
    use_truncate = any(v is not None for v in truncate.values())
    fitter = _fit_series(
        series, distributions, metrics, truncate if use_truncate else None
    )

    cols = ["name", "params", "aic", "bic", "log_likelihood", "chisquare", "ks"]
    summary = fitter.summary()
    summary = summary[[c for c in cols if c in summary.columns]]
    if select_metric in summary.columns:
        summary = summary.sort_values(select_metric)

    best_by_metric = {
        m: {"name": fit["name"], "params": _to_native(fit["params"])}
        for m, fit in fitter.best_fits.items()
    }
    best = fitter.get_best_fit(select_metric)
    return {
        "n_observations": int(len(fitter.data)),
        "select_metric": select_metric,
        "best_fit": (
            {"name": best["name"], "params": _to_native(best["params"])}
            if best is not None
            else None
        ),
        "best_by_metric": best_by_metric,
        "summary": _df_records(summary),
    }


@mcp.tool()
def sample_distribution(
    data: Optional[List[float]] = None,
    input_path: Optional[str] = None,
    column: Optional[str] = None,
    distribution: Optional[str] = None,
    distributions: Optional[List[str]] = None,
    size: int = 1000,
    zero_prop: float = 0.0,
    one_prop: float = 0.0,
    include_samples: bool = True,
    max_returned: int = 1000,
) -> Dict[str, Any]:
    """Fit data then draw random samples from the best (or chosen) distribution.

    When zero_prop or one_prop are set, a mixed sample is produced with that
    fraction of exact 0s / 1s blended into the draws (sample_mixed).

    Args:
        data / input_path / column: Data to fit (see fit_distributions).
        distribution: Force sampling from this specific fitted distribution.
        distributions: Candidate distributions for the fit (defaults to all).
        size: Number of samples to draw.
        zero_prop: Fraction of zeros to mix in (0..1).
        one_prop: Fraction of ones to mix in (0..1).
        include_samples: If True, return the draw values (capped at max_returned).
        max_returned: Maximum number of sample values to return inline.

    Returns:
        The chosen distribution, summary statistics of the draws, and optionally
        the sample values.
    """
    series = _series_from_input(data, input_path, column)
    fitter = _fit_series(series, distributions, metrics=["aic", "bic"])
    if distribution:
        fitter.select_distribution(distribution)
    chosen = fitter.selected_fit

    if zero_prop or one_prop:
        draws = np.asarray(
            fitter.sample_mixed(zero_prop=zero_prop, one_prop=one_prop, size=size)
        )
    else:
        draws = np.asarray(fitter.sample(size=size))

    result: Dict[str, Any] = {
        "distribution": chosen["name"],
        "params": _to_native(chosen["params"]),
        "size": int(draws.size),
        "statistics": {
            "mean": _to_native(np.mean(draws)),
            "std": _to_native(np.std(draws)),
            "min": _to_native(np.min(draws)),
            "max": _to_native(np.max(draws)),
        },
    }
    if include_samples:
        result["samples"] = _to_native(draws[:max_returned])
        result["samples_truncated"] = bool(draws.size > max_returned)
    return result


@mcp.tool()
def predict_pdf(
    x: List[float],
    data: Optional[List[float]] = None,
    input_path: Optional[str] = None,
    column: Optional[str] = None,
    distribution: Optional[str] = None,
    distributions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit data, then evaluate the selected distribution's PDF at the given x values.

    Args:
        x: Points at which to evaluate the probability density function.
        data / input_path / column: Data to fit (see fit_distributions).
        distribution: Force using this specific fitted distribution.
        distributions: Candidate distributions for the fit (defaults to all).

    Returns:
        The chosen distribution, the x values, and the corresponding pdf values.
    """
    series = _series_from_input(data, input_path, column)
    fitter = _fit_series(series, distributions, metrics=["aic", "bic"])
    if distribution:
        fitter.select_distribution(distribution)
    chosen = fitter.selected_fit
    pdf_values = fitter.predict(np.asarray(x, dtype=float))
    return {
        "distribution": chosen["name"],
        "params": _to_native(chosen["params"]),
        "x": _to_native(x),
        "pdf": _to_native(pdf_values),
    }


@mcp.tool()
def distribution_statistics(
    data: Optional[List[float]] = None,
    input_path: Optional[str] = None,
    column: Optional[str] = None,
    distribution: Optional[str] = None,
    distributions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare empirical data statistics against the fitted distribution.

    Returns mean, std and the 5/25/50/75/95 percentiles for both the observed
    data and the selected fitted distribution.

    Args:
        data / input_path / column: Data to fit (see fit_distributions).
        distribution: Force using this specific fitted distribution.
        distributions: Candidate distributions for the fit (defaults to all).
    """
    series = _series_from_input(data, input_path, column)
    fitter = _fit_series(series, distributions, metrics=["aic", "bic"])
    if distribution:
        fitter.select_distribution(distribution)
    stats = fitter.calculate_statistics()
    return {
        "distribution": fitter.selected_fit["name"],
        "params": _to_native(fitter.selected_fit["params"]),
        "percentile_labels": [5, 25, 50, 75, 95],
        "statistics": _to_native(stats.to_dict()),
    }


@mcp.tool()
def plot_fitted_distributions(
    output_path: str,
    data: Optional[List[float]] = None,
    input_path: Optional[str] = None,
    column: Optional[str] = None,
    distributions: Optional[List[str]] = None,
    distribution_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit data and save a PNG overlaying the fitted PDFs on the data histogram.

    Args:
        output_path: Where to write the PNG figure.
        data / input_path / column: Data to fit (see fit_distributions).
        distributions: Candidate distributions for the fit (defaults to all).
        distribution_names: Subset of fitted distributions to draw (defaults to all).

    Returns:
        The path to the saved figure.
    """
    series = _series_from_input(data, input_path, column)
    fitter = _fit_series(series, distributions, metrics=["aic", "bic"])
    out = _render_plot(lambda: fitter.plot_predictions(distribution_names), output_path)
    return {"output_path": out, "n_observations": int(len(fitter.data))}


# ---------------------------------------------------------------------------
# StochasticSimulator tools
# ---------------------------------------------------------------------------

def _build_simulator(
    freq_dist: str,
    freq_params: Sequence[Number],
    sev_dist: str,
    sev_params: Sequence[Number],
    num_sim: int,
    keep_all: bool,
    seed: int,
    correlation: Optional[float],
    copula_type: Optional[str],
    theta: float,
) -> StochasticSimulator:
    return StochasticSimulator(
        freq_dist=freq_dist,
        freq_params=_params_tuple(freq_params),
        sev_dist=sev_dist,
        sev_params=_params_tuple(sev_params),
        num_sim=num_sim,
        keep_all=keep_all,
        seed=seed,
        correlation=correlation,
        copula_type=copula_type,
        theta=theta,
    )


@mcp.tool()
def simulate_aggregate_loss(
    freq_dist: str,
    freq_params: List[float],
    sev_dist: str,
    sev_params: List[float],
    num_sim: int = 10000,
    seed: int = 1,
    correlation: Optional[float] = None,
    copula_type: Optional[str] = None,
    theta: float = 0.0,
    quantiles: Optional[List[float]] = None,
    keep_all: bool = False,
    per_occ_ded: Optional[float] = None,
    per_occ_limit: Optional[float] = None,
    agg_ded: Optional[float] = None,
    agg_limit: Optional[float] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an aggregate frequency/severity Monte Carlo simulation.

    Draws a yearly event count from the frequency distribution, a severity per
    event, and sums them into an annual aggregate loss, repeated num_sim times.
    Optionally couples frequency and severity via a copula or linear correlation,
    and applies a per-occurrence + aggregate deductible/limit layer.

    Args:
        freq_dist: Frequency distribution name (e.g. 'poisson').
        freq_params: Frequency parameters, e.g. [5].
        sev_dist: Severity distribution name (e.g. 'lognormal').
        sev_params: Severity parameters, e.g. [7, 1.2].
        num_sim: Number of simulated years.
        seed: Random seed.
        correlation: Frequency/severity correlation (optional).
        copula_type: One of gaussian/frank/gumbel/clayton (optional).
        theta: Copula theta parameter (for frank/gumbel/clayton).
        quantiles: Quantiles to report (default 0.5 0.75 0.9 0.95 0.99).
        keep_all: Keep per-event data, enabling OEP and layer treatment.
        per_occ_ded / per_occ_limit: Per-occurrence deductible / limit (need keep_all).
        agg_ded / agg_limit: Annual aggregate deductible / limit (need keep_all).
        output_path: Optional CSV path to write the full simulation output.

    Returns:
        Summary statistics, a quantile risk table (VaR/TVaR/AEP and OEP when
        keep_all), aggregate percentiles, and optional layered-loss summary.
    """
    if copula_type is not None and copula_type not in COPULA_TYPES:
        raise ValueError(f"copula_type must be one of {COPULA_TYPES}")

    layered_requested = any(
        v is not None for v in (per_occ_ded, per_occ_limit, agg_ded, agg_limit)
    )
    if layered_requested and not keep_all:
        keep_all = True  # layer treatment requires event-level data

    sim = _build_simulator(
        freq_dist, freq_params, sev_dist, sev_params,
        num_sim, keep_all, seed, correlation, copula_type, theta,
    )
    sim.gen_agg_simulations()

    qs = quantiles or [0.5, 0.75, 0.9, 0.95, 0.99]
    results_arr = np.asarray(sim.results)

    # Risk metrics: VaR/TVaR/AEP on the aggregate distribution, plus the
    # Occurrence Exceedance Probability (OEP) on the worst event per year when
    # event-level data is retained.
    quantile_table: List[Dict[str, Any]] = []
    oep_curve = None
    if keep_all and not sim.all_simulations.empty:
        max_per_year = np.sort(
            sim.all_simulations.groupby("year")["amount"].max().to_numpy()
        )
        oep_curve = max_per_year
    for q in qs:
        row = {
            "quantile": q,
            "VaR": _to_native(np.quantile(results_arr, q)),
            "TVaR": _to_native(_tvar(results_arr, q)),
            "AEP": _to_native(np.quantile(results_arr, q)),
        }
        if oep_curve is not None:
            idx = min(int(q * len(oep_curve)), len(oep_curve) - 1)
            row["OEP"] = _to_native(oep_curve[idx])
        quantile_table.append(row)

    out: Dict[str, Any] = {
        "num_sim": num_sim,
        "statistics": {
            "mean": _to_native(np.mean(results_arr)),
            "std": _to_native(np.std(results_arr)),
            "min": _to_native(np.min(results_arr)),
            "max": _to_native(np.max(results_arr)),
        },
        "aggregate_percentiles": {
            str(q): _to_native(sim.calc_agg_percentile(q * 100)) for q in qs
        },
        "quantile_table": quantile_table,
    }

    layered = None
    if layered_requested:
        layered = sim.apply_deductible_and_limit(
            per_occurrence_ded=per_occ_ded or 0.0,
            per_occurrence_limit=per_occ_limit if per_occ_limit is not None else np.inf,
            agg_ded=agg_ded or 0.0,
            agg_limit=agg_limit if agg_limit is not None else np.inf,
        )
        out["layered"] = {
            "mean_net_loss": _to_native(layered["gross_loss"].mean()),
            "net_loss_quantiles": {
                str(q): _to_native(np.quantile(layered["gross_loss"], q)) for q in qs
            },
        }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if layered is not None:
            layered.to_csv(path, index=False)
        elif keep_all:
            sim.all_simulations.to_csv(path, index=False)
        else:
            sim.results.to_frame("aggregate_loss").to_csv(path, index=False)
        out["output_path"] = str(path)

    return out


@mcp.tool()
def simulate_correlated_loss(
    corr_matrix_path: str,
    dist_list_path: str,
    num_sim: int = 10000,
    seed: int = 1,
    quantiles: Optional[List[float]] = None,
    keep_all: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a multivariate correlated simulation across lines of business.

    Reads a correlation matrix (CSV, first column = index) and a list of
    distributions (JSON: list of {dist_name, dist_type, dist_param}), generates
    correlated marginal draws via a Cholesky/Gaussian-copula percentile transform,
    and sums them into a portfolio aggregate.

    Args:
        corr_matrix_path: Path to the CSV correlation matrix.
        dist_list_path: Path to the JSON list of line-of-business distributions.
        num_sim: Number of simulations.
        seed: Random seed.
        quantiles: Quantiles to report (default 0.5 0.75 0.9 0.95 0.99).
        keep_all: Also export per-LoB marginal simulations.
        output_path: Optional CSV path for the simulation output.

    Returns:
        Summary statistics and a VaR/TVaR quantile table for the portfolio.
    """
    sim = StochasticSimulator(
        freq_dist="poisson",
        freq_params=(1,),
        sev_dist="normal",
        sev_params=(0, 1),
        num_sim=num_sim,
        keep_all=keep_all,
        seed=seed,
    )
    aggregate = sim.gen_multivariate_corr_simulations(
        corr_matrix_file=corr_matrix_path,
        dist_list_file=dist_list_path,
        gen_marginal=keep_all,
    )
    aggregate = np.asarray(aggregate)

    qs = quantiles or [0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_table = [
        {
            "quantile": q,
            "VaR": _to_native(np.quantile(aggregate, q)),
            "TVaR": _to_native(_tvar(aggregate, q)),
        }
        for q in qs
    ]

    out: Dict[str, Any] = {
        "num_sim": num_sim,
        "statistics": {
            "mean": _to_native(np.mean(aggregate)),
            "std": _to_native(np.std(aggregate)),
            "min": _to_native(np.min(aggregate)),
            "max": _to_native(np.max(aggregate)),
        },
        "quantile_table": quantile_table,
    }

    if output_path:
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if keep_all:
            dist_names = [
                d["dist_name"] for d in json.loads(Path(dist_list_path).read_text())
            ]
            df = pd.DataFrame(sim._all_simulations_data.T, columns=dist_names)
            df["aggregate"] = aggregate
            df.to_csv(path, index=False)
        else:
            pd.Series(aggregate, name="aggregate").to_frame().to_csv(path, index=False)
        out["output_path"] = str(path)

    return out


@mcp.tool()
def plot_loss_distribution(
    freq_dist: str,
    freq_params: List[float],
    sev_dist: str,
    sev_params: List[float],
    output_path: str,
    num_sim: int = 10000,
    seed: int = 1,
    bins: Optional[int] = None,
    log_scale: bool = False,
) -> Dict[str, Any]:
    """Simulate aggregate losses and save a histogram of the loss distribution.

    Args:
        freq_dist / freq_params / sev_dist / sev_params: Simulation inputs.
        output_path: Where to write the PNG figure.
        num_sim: Number of simulated years.
        seed: Random seed.
        bins: Histogram bins (defaults to sqrt(num_sim)).
        log_scale: Use a logarithmic y axis.
    """
    sim = _build_simulator(
        freq_dist, freq_params, sev_dist, sev_params,
        num_sim, False, seed, None, None, 0.0,
    )
    sim.gen_agg_simulations()
    out = _render_plot(
        lambda: sim.plot_distribution(bins=bins, log_option=log_scale), output_path
    )
    return {"output_path": out, "num_sim": num_sim}


@mcp.tool()
def plot_correlated_variables(
    freq_dist: str,
    freq_params: List[float],
    sev_dist: str,
    sev_params: List[float],
    output_path: str,
    correlation: float,
    copula_type: Optional[str] = None,
    theta: float = 0.0,
    num_sim: int = 10000,
    seed: int = 1,
) -> Dict[str, Any]:
    """Simulate correlated frequency/severity and save a scatter/KDE of the dependence.

    Args:
        freq_dist / freq_params / sev_dist / sev_params: Simulation inputs.
        output_path: Where to write the PNG figure.
        correlation: Frequency/severity correlation to induce.
        copula_type: Optional copula (gaussian/frank/gumbel/clayton).
        theta: Copula theta parameter.
        num_sim: Number of simulated years.
        seed: Random seed.
    """
    sim = _build_simulator(
        freq_dist, freq_params, sev_dist, sev_params,
        num_sim, True, seed, correlation, copula_type, theta,
    )
    sim.gen_agg_simulations()
    out = _render_plot(lambda: sim.plot_correlated_variables(), output_path)
    return {"output_path": out, "num_sim": num_sim}


# ---------------------------------------------------------------------------
# ClaimSimulator tools
# ---------------------------------------------------------------------------

def _load_policies(
    policies: Optional[List[Dict[str, Any]]],
    policies_path: Optional[str],
) -> pd.DataFrame:
    """Load a policy book from inline records or a CSV, coercing param columns."""
    if policies is not None and policies_path:
        raise ValueError("Provide either `policies` or `policies_path`, not both.")
    if policies is not None:
        df = pd.DataFrame(policies)
    elif policies_path:
        df = pd.read_csv(policies_path)
    else:
        raise ValueError("Supply a policy book via `policies` or `policies_path`.")

    for col in ("freq_params", "sev_params"):
        if col in df.columns:
            df[col] = df[col].apply(_coerce_param_cell)
    return df


@mcp.tool()
def simulate_claims(
    policies: Optional[List[Dict[str, Any]]] = None,
    policies_path: Optional[str] = None,
    seed: int = 42,
    correlation: Optional[float] = None,
    copula_type: Optional[str] = None,
    copula_param: float = 0.0,
    assign_dates: bool = False,
    nhpp_lambda0: float = 10.0,
    nhpp_alpha: float = 0.5,
    nhpp_phase: float = 0.0,
    nhpp_T: float = 1.0,
    ldfs: Optional[Dict[str, float]] = None,
    dev_volatility: float = 0.1,
    dev_cumulative_factor: float = 1.0,
    output_path: Optional[str] = None,
    max_returned: int = 200,
) -> Dict[str, Any]:
    """Simulate individual claims from a policy book.

    Each policy needs: policy_id, freq_dist, freq_params, sev_dist, sev_params,
    start_date, end_date. The tool simulates claim counts and severities per policy,
    can assign incurred dates via a non-homogeneous Poisson process, and can build a
    loss-development triangle from base loss-development factors (LDFs).

    Args:
        policies: Inline list of policy dicts. Mutually exclusive with policies_path.
        policies_path: Path to a policies CSV.
        seed: Random seed.
        correlation: Frequency/severity correlation (optional).
        copula_type: Copula type (gaussian/frank/gumbel/clayton, optional).
        copula_param: Copula parameter.
        assign_dates: Assign incurred dates with an NHPP (required before LDFs).
        nhpp_lambda0 / nhpp_alpha / nhpp_phase / nhpp_T: NHPP parameters.
        ldfs: Base loss-development factors keyed by development month, e.g.
            {"12": 2.5, "24": 1.5}. Requires assign_dates=True.
        dev_volatility: Per-claim LDF noise.
        dev_cumulative_factor: Tail cumulative factor for development.
        output_path: Optional CSV path for the claims (development saved alongside).
        max_returned: Maximum claim rows to return inline.

    Returns:
        Claim counts, loss totals, a preview of the claims, and (if requested)
        development-triangle summary.
    """
    if ldfs is not None and not assign_dates:
        raise ValueError("`ldfs` requires `assign_dates=True` so claims have incurred dates.")

    policies_df = _load_policies(policies, policies_path)
    sim = ClaimSimulator(
        policies_df=policies_df,
        random_seed=seed,
        correlation=correlation,
        copula_type=copula_type,
        copula_param=copula_param,
    )
    sim.simulate_claims()

    if assign_dates:
        sim.simulate_dates_nhpp(
            lambda0=nhpp_lambda0, alpha=nhpp_alpha, phase=nhpp_phase, T=nhpp_T
        )

    claims = sim.claim_data
    if claims is None or claims.empty:
        return {"n_claims": 0, "claims": [], "message": "No claims generated."}

    loss_col = "ultimate_loss" if "ultimate_loss" in claims.columns else "amount"
    out: Dict[str, Any] = {
        "n_claims": int(len(claims)),
        "n_policies": int(claims["policy_id"].nunique()),
        "loss_column": loss_col,
        "total_loss": _to_native(claims[loss_col].sum()),
        "mean_loss": _to_native(claims[loss_col].mean()),
        "max_loss": _to_native(claims[loss_col].max()),
        "claims": _df_records(claims, max_rows=max_returned),
        "claims_truncated": bool(len(claims) > max_returned),
    }

    development_df = None
    if ldfs is not None:
        base_ldfs = {int(k): float(v) for k, v in ldfs.items()}
        sim.simulate_claim_development(
            base_LDFs=base_ldfs,
            volatility=dev_volatility,
            cumulative_factor=dev_cumulative_factor,
        )
        development_df = sim.claim_development
        out["development"] = {
            "n_records": int(len(development_df)),
            "n_accident_years": int(development_df["accident_year"].nunique()),
            "preview": _df_records(development_df, max_rows=max_returned),
        }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        claims.to_csv(path, index=False)
        out["output_path"] = str(path)
        if development_df is not None:
            dev_path = path.with_name(path.stem + "_development.csv")
            development_df.to_csv(dev_path, index=False)
            out["development_output_path"] = str(dev_path)

    return out


@mcp.tool()
def build_claim_development(
    ldfs: Dict[str, float],
    claims: Optional[List[Dict[str, Any]]] = None,
    input_path: Optional[str] = None,
    volatility: float = 0.1,
    cumulative_factor: float = 1.0,
    seed: int = 42,
    triangle: bool = False,
    output_path: Optional[str] = None,
    max_returned: int = 200,
) -> Dict[str, Any]:
    """Build a loss-development triangle from existing claims.

    Claims must include columns: event_id, policy_id, ultimate_loss, incurred_date.
    Applies stochastic loss-development factors to spread each claim's ultimate loss
    across development months.

    Args:
        ldfs: Base loss-development factors keyed by development month, e.g.
            {"12": 2.5, "24": 1.5}.
        claims: Inline list of claim dicts. Mutually exclusive with input_path.
        input_path: Path to a claims CSV.
        volatility: Per-claim LDF noise.
        cumulative_factor: Tail cumulative factor for development.
        seed: Random seed.
        triangle: If True, also return an accident-year x development-month pivot.
        output_path: Optional CSV path for the development data.
        max_returned: Maximum development rows to return inline.

    Returns:
        Development record counts, a preview, and optionally the pivoted triangle.
    """
    if claims is not None and input_path:
        raise ValueError("Provide either `claims` or `input_path`, not both.")
    if claims is not None:
        df = pd.DataFrame(claims)
    elif input_path:
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Supply claims via `claims` or `input_path`.")

    required = {"event_id", "policy_id", "ultimate_loss", "incurred_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Claims are missing required columns: {sorted(missing)}.")

    # Build a stub policy book so we can reuse the package's development logic.
    stub_policies = pd.DataFrame(
        [
            {
                "policy_id": pid,
                "freq_dist": "poisson",
                "freq_params": (1,),
                "sev_dist": "normal",
                "sev_params": (0.0, 1.0),
                "start_date": "2000-01-01",
                "end_date": "2000-12-31",
            }
            for pid in df["policy_id"].dropna().unique()
        ]
    )
    sim = ClaimSimulator(policies_df=stub_policies, random_seed=seed)
    sim.claim_data = df.copy()
    sim.claim_data["incurred_date"] = pd.to_datetime(sim.claim_data["incurred_date"])

    base_ldfs = {int(k): float(v) for k, v in ldfs.items()}
    sim.simulate_claim_development(
        base_LDFs=base_ldfs, volatility=volatility, cumulative_factor=cumulative_factor
    )
    dev = sim.claim_development

    out: Dict[str, Any] = {
        "n_records": int(len(dev)),
        "n_accident_years": int(dev["accident_year"].nunique()),
        "n_development_months": int(dev["development_month"].nunique()),
        "preview": _df_records(dev, max_rows=max_returned),
    }

    if triangle:
        tri = dev.pivot_table(
            index="accident_year",
            columns="development_month",
            values="incurred_loss",
            aggfunc="sum",
        ).round(2)
        out["triangle"] = {
            "development_months": _to_native(list(tri.columns)),
            "rows": {
                str(ay): _to_native(list(row.values)) for ay, row in tri.iterrows()
            },
        }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dev.to_csv(path, index=False)
        out["output_path"] = str(path)

    return out


# ---------------------------------------------------------------------------
# Plot rendering helper
# ---------------------------------------------------------------------------

def _render_plot(plot_call, output_path: str) -> str:
    """Run a plotting callable (which ends in plt.show()) and save the figure."""
    plt.close("all")
    plot_call()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return str(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="actsim-mcp",
        description="Run the actsim MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio, for Claude Desktop).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    args = parser.parse_args(argv)

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
