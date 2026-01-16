# Random Networks Visualization

Interactive visualization tools for exploring random network models. This project generates animated HTML visualizations showing network growth and degree distributions for two fundamental random graph models:

- **Erdos-Renyi (ER) Model**: Random graphs where edges are created with fixed probability p
- **Barabasi-Albert (BA) Model**: Scale-free networks grown via preferential attachment

## Features

- Animated network growth visualization with interactive slider controls
- Real-time degree distribution plots
- Theoretical distribution overlays (Poisson for ER, power-law for BA)
- Dark theme professional styling
- Exportable standalone HTML files

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (recommended)

```bash
uv sync
```

## Usage

### Generate Barabasi-Albert Network Visualization

```bash
python src/random_networks/ba_network_viz.py
```

This generates `ba_network.html` showing scale-free network growth with preferential attachment. The degree distribution follows a power-law P(k) ~ k^(-3).

### Generate Erdos-Renyi Network Visualization

```bash
python src/random_networks/er_network_viz.py
```

This generates `er_network.html` showing random network growth. The degree distribution follows a Poisson distribution.

## Output

Generated HTML files are saved in the `results/` folder. Open them in any web browser to interact with the visualizations:

- Use the slider to step through network growth
- Use playback controls (Pause, 1x, 2x, 4x) for animation
- Hover over nodes to see degree information

## Project Structure

```
random-networks/
├── src/random_networks/     # Source code
│   ├── ba_network_viz.py    # Barabasi-Albert visualization
│   └── er_network_viz.py    # Erdos-Renyi visualization
├── results/                 # Generated HTML visualizations
├── requirements.txt         # Python dependencies
└── pyproject.toml          # Project configuration
```

## Dependencies

- networkx: Graph creation and analysis
- numpy: Numerical computations
- plotly: Interactive visualizations
- scipy: Statistical distributions

## License

MIT
