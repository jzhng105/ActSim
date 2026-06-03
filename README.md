# ActSim

[![PyPI version](https://badge.fury.io/py/ActSim.svg)](https://badge.fury.io/py/ActSim)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for actuarial risk modeling and simulation.

## Features

- **Frequency-Severity Simulation**: Monte Carlo modeling of aggregate losses using configurable frequency and severity distributions (Poisson, lognormal, gamma, Pareto, and more)
- **Synthetic Claim Generation**: Produce policy and claim level datasets for testing, benchmarking, and stress-testing actuarial workflows when real data is limited or restricted
- **Distribution Fitting**: Fit, compare, and diagnose candidate distributions against observed data with built-in goodness-of-fit testing
- **Claim Development**: Project claims through time using loss development factors, with native support for accident-year and development-age structures
- **Risk Metrics**: Compute mean, percentiles, VaR, TVaR, AEP, and OEP from simulated loss distributions
- **Chainladder Integration**: Convert simulated claims into triangle format for reserving, IBNR estimation, and ultimate loss projection using the `chainladder` package
- **Reproducibility**: Seed-controlled simulations and transparent validation tools designed to meet open-source and CAS review standards
- **Python-Native**: Built on ActStats, NumPy, pandas, and SciPy for seamless integration with the modern data science and actuarial analytics stack
- **MCP Server**: Drive the entire toolkit from an LLM client (Claude Desktop/Code) via the built-in [Model Context Protocol server](docs/mcp_server.md)

## Installation

### From PyPI (recommended)

```bash
pip install ActSim
```

### From Source

```bash
git clone https://github.com/jzhng105/ActSim.git
cd ActSim
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/jzhng105/ActSim.git
cd ActSim
pip install -e .[dev]
```

## Quick Start

```python
from ActSim import load_config, DistributionFitter
from actstats import actuarial as act

# Load configuration
config = load_config()

sev_data = act.lognormal(0.5,0.2).rvs(size=10000)

#############################
###### Fit Severity #########
#############################
# User specifies distributions and metrics 
distribution_names = config.distributions['severity']
metrics = config.metrics

sev_fitter = actfitter(sev_data, distributions=distribution_names, metrics=metrics)
sev_fitter.fit()
sev_fitter.best_fits
sev_fitter.selected_fit
```

## MCP Server

ActSim can run as a [Model Context Protocol](https://modelcontextprotocol.io)
server, exposing every public function as a tool for LLM clients such as Claude
Desktop and Claude Code.

```bash
pip install "actsim[mcp]"   # install with the MCP extra
actsim-mcp                  # launch the server (stdio transport)
```

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "actsim": { "command": "actsim-mcp" }
  }
}
```

See the [MCP server guide](docs/mcp_server.md) for the full tool reference.

## Documentation

- [User Guide](docs/user_guide.md) - Getting started and basic usage
- [MCP Server](docs/mcp_server.md) - Run ActSim as an MCP server for LLM clients
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Examples](docs/examples.md) - Code examples and tutorials
- [Development](docs/development.md) - Contributing and development guidelines

## Features in Detail

### Configuration Management
```python
# Initialize fitter with config file
config = load_config()

```

### Fit Distributions
```python
# ---------------------------------------------
# Import required modules
# ---------------------------------------------
from ActSim import load_config, DistributionFitter
from actstats import actuarial as act

# ---------------------------------------------
# 1. Generate Example Data
# ---------------------------------------------
# Severity data: Using lognormal distribution with mu=0.5 and sigma=0.2
sev_data = act.lognormal(0.5, 0.2).rvs(size=10000)

# Frequency data: Using Poisson distribution with λ=10
freq_data = act.poisson.rvs(10, 1000)

# ---------------------------------------------
# 2. Load Configuration
# ---------------------------------------------
# This loads distribution lists and metrics from the ActSim config file
config = load_config()

# ---------------------------------------------
# 3. Fit Severity Distributions
# ---------------------------------------------
# Get severity distributions and metrics from config
distribution_names = config.distributions['severity']
metrics = config.metrics

# Initialize severity fitter
sev_fitter = DistributionFitter(sev_data, distributions=distribution_names, metrics=metrics)

# Perform fitting
sev_fitter.fit()

# View best fits and selected distribution
print("Best fits:", sev_fitter.best_fits)
print("Selected fit:", sev_fitter.selected_fit)
print("Selected distribution object:", sev_fitter.get_selected_dist())

# Manually selecting a distribution (example: 'uniform')
sev_fitter.select_distribution('uniform')
selected_fit = sev_fitter.selected_fit

# Print details of the selected fit
print("Selected fitting distribution:", selected_fit['name'])
print("Parameters:", selected_fit['params'])
print("AIC:", selected_fit['aic'])
print("BIC:", selected_fit['bic'])

# Calculate statistics for severity
sev_fitter.calculate_statistics()

# Plot predictions
sev_fitter.plot_predictions()

# Print summary report
sev_fitter.summary()

# ---------------------------------------------
# 4. Generate Samples from Severity Fit
# ---------------------------------------------
samples = sev_fitter.sample(size=10)
print("Generated samples:", samples)

# Generate mixed samples (e.g., weighted combinations)
samples = sev_fitter.sample_mixed(0.1, 0.1, size=10)
print("Generated samples:", samples)

# ---------------------------------------------
# 5. Fit Frequency Distributions
# ---------------------------------------------
distribution_names = config.distributions['frequency']
metrics = config.metrics

# Initialize frequency fitter
freq_fitter = DistributionFitter(freq_data, distributions=distribution_names, metrics=metrics)

# Show available frequency distributions
print("Frequency distributions:", freq_fitter.distributions)

# Perform fitting
freq_fitter.fit()

# View best fits and summary
print("Frequency best fits:", freq_fitter.best_fits)
print("Frequency selected fit:", freq_fitter.selected_fit)
freq_fitter.summary()
```

### Stochastic Simulation

```python
#####################################
###### Stochastic Simulation ########
#####################################
# ---------------------------------------------
# 1. Import Required Modules
# ---------------------------------------------
from ActSim import StochasticSimulator
from actstats import actuarial as act

# ---------------------------------------------
# 2. Define Frequency and Severity Distributions
# ---------------------------------------------
# Frequency distribution: Poisson with λ=10
freq_dist = 'poisson'
freq_params = (10,)

# Severity distribution: Lognormal with mu=10, sigma=0.5
sev_dist = 'lognormal'
sev_params = (10, 0.5)

# Preview quantile (e.g., 80th percentile of Poisson)
quantile_80 = act.poisson.ppf(0.8, 10)
print("80th percentile of Poisson(10):", quantile_80)

# ---------------------------------------------
# 3. Initialize Simulator with Different Levels of Complexity
# ---------------------------------------------

# With copula
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234, 0.6, 'frank', 0.6)

# with linear correlation
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234, 0.6)

# Without copula or linear correlation
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234)

# ---------------------------------------------
# 4. Generate Simulated Aggregate Losses
# ---------------------------------------------
simulations = simulator.gen_agg_simulations()

# Access full simulation DataFrame
print("All simulations preview:")
print(simulator.all_simulations.head())

# ---------------------------------------------
# 5. Analyze Simulation Results
# ---------------------------------------------

# Calculate aggregate percentile (e.g., 99.2%)
percentile_99_2 = simulator.calc_agg_percentile(99.2)
print("99.2% Aggregate Loss Percentile:", percentile_99_2)

# Plot loss distribution histogram
simulator.plot_distribution()

# Show simulation mean
print("Mean simulated loss:", simulator.results.mean())

# If copula is used, plot frequency-severity correlation structure
simulator.plot_correlated_variables()

# Summary statistics and shape diagnostics
simulator.analyze_results()

# ---------------------------------------------
# 6. Apply Deductibles and Limits
# ---------------------------------------------
# Apply per occurrence deductible of 1,000
# Occurrence limit of 10,000
# Annual aggregate deductible of 100,000
# Annual aggregate limit of 300,000
gross_loss = simulator.apply_deductible_and_limit(1000, 10000, 100000, 300000)


# Assign processed loss to expected structure for reporting
gross_loss['amount'] = gross_loss['gross_loss']

# Re-analyze results based on capped/layered gross loss
simulator.analyze_results(all_simulations=gross_loss)

# ---------------------------------------------
# 7. Export Simulated Data to CSV
# ---------------------------------------------
simulator.all_simulations
```

### Correlated Mutivariate Distribution Simulation

Sample correlation matrix csv file
```csv
Correlation Matrix, LoB1, LoB2, LoB3, LoB4, LoB5
LoB1, 1, 0.5, 0.5, 0.3, 0.2
LoB2, 0.5, 1, 0.5, 0.7, 0.3 
LoB3, 0.5, 0.5, 1, 0.2, 0.5
LoB4, 0.3, 0.7, 0.2, 1, 0.3
LoB5, 0.2, 0.3, 0.5, 0.3, 1
```

Sample multi-line distribution json file
```json
[
    {
        "index": 1,
        "dist_name": "LoB1",
        "dist_type": "gamma",
        "dist_param": [2, 1]
    },
    {
        "index": 2,
        "dist_name": "LoB2",
        "dist_type": "lognormal",
        "dist_param": [2, 1]
    },
    {
        "index": 3,
        "dist_name": "LoB3",
        "dist_type": "gamma",
        "dist_param": [3, 1]
    },
    {
        "index": 4,
        "dist_name": "LoB4",
        "dist_type": "lognormal",
        "dist_param": [3, 2]
    },
    {
        "index": 5,
        "dist_name": "LoB5",
        "dist_type": "gamma",
        "dist_param": [4, 2]
    }
]
```

```python
import pandas as pd
from ActSim import StochasticSimulator

##### Generate correlated mutivariate distribution
corr_matrix_file = 'examples/correlated_sim/corr_matrix.csv'
dist_list_file = 'examples/correlated_sim/dist_list.json'
simulator = StochasticSimulator("normal", [1,0], "normal",[1,0], 100000, True, 1234) # placeholder parameters for the simulator
simulator.gen_multivariate_corr_simulations(corr_matrix_file, dist_list_file, True)
simulator._all_simulations_data
data = pd.DataFrame(simulator._all_simulations_data)
data_t = data.transpose()
# Compute correlation matrix
correlation_matrix = data_t.corr()
print(correlation_matrix)
```
### Synthetic Claim Simulation

```python
##########################################
###### Synthetic Claim Simulation ########
##########################################
import pandas as pd
import numpy as np
from ActSim import ClaimSimulator

# Simulate policy characteristics
policies = pd.DataFrame({
    'policy_id': range(1, 101),
    'freq_dist': 'poisson',
    'freq_params': list(zip(np.random.uniform(0.6, 0.8, 100).round(2),)), 
    'sev_dist': 'lognormal',
    'sev_params': list(zip(np.random.uniform(0.8, 1.2, 100).round(2), np.random.uniform(0.3, 0.7, 100).round(2))),
    'start_date': pd.Timestamp('2023-01-01'),
    'end_date': pd.Timestamp('2023-12-31'),
})

# Instantiate the ClaimSimulator with input policies and np random seed 42
claim_sim = ClaimSimulator(policies, 42)

# Access the processed policy DataFrame
claim_sim.policies

# Run the claim simulation (frequency × severity) for all policy groups
claim_sim.simulate_claims()

# Access the resulting simulated claim records
claim_sim.claim_data

# Set parameters for the non-homogeneous Poisson process (NHPP) for date simulation
lambda0 = 10     # Baseline intensity
alpha = 0.5      # Seasonality amplitude
phase = 0        # Phase shift of the seasonality
T = 1            # Duration of the exposure in years

# Simulate claim occurrence dates using a seasonal NHPP
claim_sim.simulate_dates_nhpp(lambda0, alpha, phase, T)

# Define base loss development factors (LDFs) by development month
base_LDFs = {
    0: 2,     # Initial LDF at 0 months
    3: 1.5,   # LDF at 3 months
    6: 1.2,
    9: 1.1,
    12: 1.05,
    15: 1.02,
    18: 1.00  # Ultimate LDF at 18 months
}

volatility = 0.1      # Standard deviation for stochastic fluctuation in LDFs
tail_factor = 1.0     # No additional tail development (fully developed at 18 months)

# Simulate the claim development triangles based on LDFs and apply stochastic volatility
claim_sim.simulate_claim_development(base_LDFs, volatility, tail_factor)

# Access the simulated claim development triangle or long-format development data
claim_sim.claim_development

# Access updated policies (could include mappings to simulated claims)
claim_sim.policies

# Save the simulated claim development data to a file (replace with actual path)
claim_sim.save_claim_development('sample_file_path')

```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/jzhng105/ActSim.git
cd ActSim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ActSim in your research, please cite:

```bibtex
@software{ActSim2025,
  title={ActSim: A Python package for actuarial risk modeling and simulation},
  author={Juntao Zhang},
  year={2025},
  url={https://github.com/jzhng105/ActSim}
}
```

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/jzhng105/ActSim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jzhng105/ActSim/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
