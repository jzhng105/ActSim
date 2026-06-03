# ActSim MCP Server

ActSim ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that exposes the full actuarial toolkit as tools an LLM client (Claude
Desktop, Claude Code, or any MCP-compatible host) can call directly. The server
wraps every public capability of the package: distribution fitting, sampling,
aggregate and correlated Monte-Carlo simulation, claim and loss-development
simulation, plotting, and configuration access.

## Installation

The MCP server depends on the optional `mcp` package. Install actsim with the
`mcp` extra:

```bash
pip install "actsim[mcp]"
# or, from a checkout
pip install -e ".[mcp]"
```

## Running

The server is registered as a console script:

```bash
actsim-mcp                       # stdio transport (default; for Claude Desktop)
actsim-mcp --transport sse --host 127.0.0.1 --port 8000
python -m actsim.mcp_server      # equivalent module form
```

### Claude Desktop configuration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "actsim": {
      "command": "actsim-mcp"
    }
  }
}
```

## Tools

Data tools accept input **inline** (a JSON list/records) or via a **file path**
(`input_path` / `policies_path`, CSV or JSON). Numeric outputs are returned as
JSON; plot tools render a PNG to `output_path` and return its path.

### Discovery & configuration

| Tool | Description |
| --- | --- |
| `version` | Installed actsim version and capability catalogue. |
| `list_distributions` | Available distributions, fit metrics, and copulas. |
| `get_config` | Load the YAML configuration (default or custom path). |
| `update_config` | Update and persist keys in a YAML config file. |

### Distribution fitting (`DistributionFitter`)

| Tool | Wraps |
| --- | --- |
| `fit_distributions` | `truncate_data`, `fit`, `select_best_fit`, `get_best_fit`, `summary`, and all goodness-of-fit metrics (aic, bic, log_likelihood, chisquare, ks). |
| `sample_distribution` | `select_distribution`, `sample`, `sample_mixed`. |
| `predict_pdf` | `predict` (PDF evaluation at chosen points). |
| `distribution_statistics` | `calculate_statistics` (data vs. fitted comparison). |
| `plot_fitted_distributions` | `plot_predictions` → PNG. |

### Aggregate simulation (`StochasticSimulator`)

| Tool | Wraps |
| --- | --- |
| `simulate_aggregate_loss` | `gen_agg_simulations`, `results`, `all_simulations`, `calc_agg_percentile`, VaR/TVaR/OEP/AEP analysis, and `apply_deductible_and_limit` (per-occurrence + aggregate layer). Supports copula / linear correlation. |
| `simulate_correlated_loss` | `gen_multivariate_corr_simulations` (correlation matrix + line-of-business distribution list). |
| `plot_loss_distribution` | `plot_distribution` → PNG. |
| `plot_correlated_variables` | `plot_correlated_variables` → PNG. |

### Claim simulation (`ClaimSimulator`)

| Tool | Wraps |
| --- | --- |
| `simulate_claims` | `group_policies`, `simulate_claims`, `simulate_dates_nhpp` (NHPP incurred dates), `simulate_claim_development` (loss-development triangle). |
| `build_claim_development` | `simulate_claim_development` from an existing claims table, with optional accident-year × development-month triangle pivot. |

## Examples

Fit distributions to inline data:

```json
// tool: fit_distributions
{
  "data": [1.2, 0.9, 2.1, 1.7, 0.6, 3.4],
  "distributions": ["lognormal", "gamma", "normal"],
  "select_metric": "aic"
}
```

Run an aggregate simulation with a per-occurrence and aggregate layer:

```json
// tool: simulate_aggregate_loss
{
  "freq_dist": "poisson",
  "freq_params": [5],
  "sev_dist": "lognormal",
  "sev_params": [7, 1.2],
  "num_sim": 10000,
  "keep_all": true,
  "per_occ_ded": 1000,
  "per_occ_limit": 100000,
  "agg_ded": 0,
  "agg_limit": 5000000
}
```

Simulate a policy book into claims, assign incurred dates, and build a triangle:

```json
// tool: simulate_claims
{
  "policies": [
    {"policy_id": "P1", "freq_dist": "poisson", "freq_params": [3],
     "sev_dist": "lognormal", "sev_params": [7, 1.0],
     "start_date": "2020-01-01", "end_date": "2020-12-31"}
  ],
  "assign_dates": true,
  "ldfs": {"12": 2.5, "24": 1.5, "36": 1.1}
}
```
