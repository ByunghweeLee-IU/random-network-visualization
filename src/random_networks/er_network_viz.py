"""
Random Network Visualization using Erdős-Rényi Model

Generates an interactive HTML visualization showing network growth from 0 to 200 nodes.
The ER model connects node pairs with a fixed probability p, producing a Poisson degree distribution.
"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random
from scipy import stats


def generate_er_network_history(n_final=200, p=0.02, seed=42):
    """
    Generate ER network growth history step by step.

    In the G(n,p) model, we add nodes one at a time and connect each new node
    to existing nodes with probability p.

    Args:
        n_final: Final number of nodes
        p: Probability of edge between any two nodes
        seed: Random seed for reproducibility

    Returns:
        List of (nodes, edges) tuples for each step
    """
    random.seed(seed)
    np.random.seed(seed)

    history = []

    # Start with initial node
    G = nx.Graph()
    G.add_node(0)
    history.append((list(G.nodes()), list(G.edges())))

    for new_node in range(1, n_final):
        G.add_node(new_node)

        # Connect to each existing node with probability p
        for existing_node in range(new_node):
            if random.random() < p:
                G.add_edge(new_node, existing_node)

        history.append((list(G.nodes()), list(G.edges())))

    return history


def compute_layout_for_all_nodes(n_final, history, seed=42):
    """
    Compute a stable layout for all nodes using the final network state.
    This ensures nodes don't jump around as the slider moves.
    """
    # Create the final graph
    final_nodes, final_edges = history[-1]
    G_final = nx.Graph()
    G_final.add_nodes_from(final_nodes)
    G_final.add_edges_from(final_edges)

    # Compute layout for final graph
    pos = nx.spring_layout(G_final, k=1.5/np.sqrt(n_final), iterations=100, seed=seed)

    return pos


def compute_logbinned_degree_distribution(degrees, num_bins=20):
    """
    Compute log-binned degree distribution for log-log plotting.

    Uses logarithmic binning where bin widths increase geometrically.

    Args:
        degrees: List of node degrees
        num_bins: Number of logarithmic bins

    Returns:
        bin_centers: Center of each bin (geometric mean)
        probabilities: Probability density P(k) for each bin
    """
    if len(degrees) == 0 or max(degrees) == 0:
        return [], []

    degrees = np.array(degrees, dtype=float)
    min_deg = max(1, min(degrees))
    max_deg = max(degrees)

    if min_deg == max_deg:
        return [min_deg], [1.0]

    # Create logarithmically spaced bin edges
    log_min = np.log10(min_deg)
    log_max = np.log10(max_deg * 1.1)
    bin_edges = np.logspace(log_min, log_max, num_bins + 1)

    bin_centers = []
    probabilities = []
    n_total = len(degrees)

    for i in range(len(bin_edges) - 1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]

        # Count degrees in this bin [left, right)
        mask = (degrees >= left_edge) & (degrees < right_edge)
        count = np.sum(mask)

        if count > 0:
            # Geometric mean for bin center
            bin_center = np.sqrt(left_edge * right_edge)

            # Linear bin width for probability density
            bin_width = right_edge - left_edge

            # Probability density
            prob_density = count / (n_total * bin_width)

            bin_centers.append(bin_center)
            probabilities.append(prob_density)

    return bin_centers, probabilities


def compute_poisson_theory(n_nodes, p, k_range):
    """
    Compute theoretical Poisson distribution for ER network.

    In ER networks, the degree distribution follows a Poisson distribution
    with mean lambda = (n-1) * p.

    Args:
        n_nodes: Number of nodes in the network
        p: Edge probability
        k_range: Range of degrees to compute

    Returns:
        k_values: Degree values
        p_values: Probability P(k) for each degree
    """
    # Mean degree in ER network
    lambda_mean = (n_nodes - 1) * p

    k_values = np.array(k_range)
    # Poisson PMF: P(k) = (lambda^k * e^-lambda) / k!
    p_values = stats.poisson.pmf(k_values.astype(int), lambda_mean)

    return k_values, p_values


def create_visualization(history, pos, p, output_file="er_network.html", frame_step=5):
    """
    Create interactive Plotly visualization with slider and degree distribution plot.
    Professional dark theme design.

    Args:
        history: List of (nodes, edges) tuples
        pos: Node positions dict
        p: Edge probability (for theoretical distribution)
        output_file: Output HTML filename
        frame_step: Sample every nth step to reduce file size
    """
    # Sample frames to reduce file size
    sampled_indices = list(range(0, len(history), frame_step))
    if sampled_indices[-1] != len(history) - 1:
        sampled_indices.append(len(history) - 1)

    n_steps = len(sampled_indices)

    # Precompute node degrees for each step
    degrees_history = []
    for nodes, edges in history:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        degrees_history.append(dict(G.degree()))

    # Color scheme - professional dark theme (green accent for ER)
    BG_COLOR = '#1a1a2e'
    PLOT_BG = '#16213e'
    TEXT_COLOR = '#eaeaea'
    GRID_COLOR = '#2a2a4a'
    EDGE_COLOR = 'rgba(100, 180, 140, 0.4)'
    ACCENT_COLOR = '#00d9a5'  # Green for ER (vs red for BA)
    DATA_COLOR = '#4cc9f0'

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.12,
        subplot_titles=("", "")
    )

    # Create frames for sampled steps
    frames = []
    slider_steps = []

    # Get max degree for consistent colorbar scaling
    max_degree_overall = max(max(degrees_history[i].values()) if degrees_history[i].values() else 1
                            for i in sampled_indices)

    for frame_idx, step in enumerate(sampled_indices):
        nodes, edges = history[step]
        degrees = degrees_history[step]
        degree_list = list(degrees.values())

        # Edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.6, color=EDGE_COLOR),
            hoverinfo='none',
            xaxis='x1', yaxis='y1'
        )

        # Node traces
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_sizes = [max(4, min(degrees[n] * 2.5, 50)) for n in nodes]
        node_colors = [degrees[n] for n in nodes]
        node_text = [f"Node {n}<br>Degree: {degrees[n]}" for n in nodes]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',  # Different colorscale for ER
                cmin=0,
                cmax=max_degree_overall,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Degree",
                        font=dict(size=14, color=TEXT_COLOR)
                    ),
                    thickness=15,
                    len=0.6,
                    x=0.52,
                    y=0.5,
                    tickfont=dict(size=11, color=TEXT_COLOR),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    outlinewidth=0
                ),
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            xaxis='x1', yaxis='y1'
        )

        # Degree distribution trace (histogram style for Poisson)
        if degree_list and max(degree_list) > 0:
            # For ER networks, use simple histogram (not log-binned)
            unique_degrees, counts = np.unique(degree_list, return_counts=True)
            probabilities = counts / len(degree_list)
        else:
            unique_degrees = [0]
            probabilities = [1]

        degree_trace = go.Scatter(
            x=unique_degrees,
            y=probabilities,
            mode='markers',
            marker=dict(
                size=10,
                color=DATA_COLOR,
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name='Empirical',
            hovertemplate='k = %{x}<br>P(k) = %{y:.3f}<extra></extra>',
            xaxis='x2', yaxis='y2'
        )

        # Theoretical Poisson line trace
        n_nodes = len(nodes)
        if n_nodes > 1:
            k_max = max(max(degree_list) if degree_list else 1, 10)
            k_theory = np.arange(0, k_max + 5)
            k_theory_plot, p_theory = compute_poisson_theory(n_nodes, p, k_theory)
            # Filter out very small probabilities for cleaner display
            mask = p_theory > 1e-6
            k_theory_plot = k_theory_plot[mask]
            p_theory = p_theory[mask]
        else:
            k_theory_plot = [0, 1]
            p_theory = [1, 0]

        theory_trace = go.Scatter(
            x=k_theory_plot,
            y=p_theory,
            mode='lines',
            line=dict(color=ACCENT_COLOR, width=2.5),
            name='Poisson',
            hoverinfo='skip',
            xaxis='x2', yaxis='y2'
        )

        frame = go.Frame(
            data=[edge_trace, node_trace, degree_trace, theory_trace],
            name=str(frame_idx),
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"<b>N = {len(nodes)}</b>  |  <b>E = {len(edges)}</b>  |  <b>p = {p}</b>",
                        xref="paper", yref="paper",
                        x=0.26, y=1.02,
                        showarrow=False,
                        font=dict(size=16, color=TEXT_COLOR),
                        bgcolor='rgba(0,0,0,0)'
                    )
                ]
            )
        )
        frames.append(frame)

        # Slider step
        show_label = (step + 1) % 50 == 0 or step == 0 or step == len(history) - 1
        slider_step = dict(
            args=[[str(frame_idx)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0)
            )],
            label=str(step + 1) if show_label else "",
            method="animate"
        )
        slider_steps.append(slider_step)

    # Initial data (first sampled frame)
    first_step = sampled_indices[0]
    nodes, edges = history[first_step]
    degrees = degrees_history[first_step]
    degree_list = list(degrees.values())

    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Add initial traces to subplots
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.6, color=EDGE_COLOR),
        hoverinfo='none',
        showlegend=False
    ), row=1, col=1)

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_sizes = [max(4, min(degrees[n] * 2.5, 50)) for n in nodes]
    node_colors = [degrees[n] for n in nodes]
    node_text = [f"Node {n}<br>Degree: {degrees[n]}" for n in nodes]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        showlegend=False,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            cmin=0,
            cmax=max_degree_overall,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Degree",
                    font=dict(size=14, color=TEXT_COLOR)
                ),
                thickness=15,
                len=0.6,
                x=0.52,
                y=0.5,
                tickfont=dict(size=11, color=TEXT_COLOR),
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
                outlinewidth=0
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        )
    ), row=1, col=1)

    # Initial degree distribution
    if degree_list and max(degree_list) > 0:
        unique_degrees, counts = np.unique(degree_list, return_counts=True)
        probabilities = counts / len(degree_list)
    else:
        unique_degrees = [0]
        probabilities = [1]

    fig.add_trace(go.Scatter(
        x=unique_degrees,
        y=probabilities,
        mode='markers',
        marker=dict(
            size=10,
            color=DATA_COLOR,
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        name='Empirical',
        hovertemplate='k = %{x}<br>P(k) = %{y:.3f}<extra></extra>'
    ), row=1, col=2)

    # Add theoretical line
    n_nodes = len(nodes)
    if n_nodes > 1:
        k_max = max(max(degree_list) if degree_list else 1, 10)
        k_theory = np.arange(0, k_max + 5)
        k_theory_plot, p_theory = compute_poisson_theory(n_nodes, p, k_theory)
        mask = p_theory > 1e-6
        k_theory_plot = k_theory_plot[mask]
        p_theory = p_theory[mask]
    else:
        k_theory_plot = [0, 1]
        p_theory = [1, 0]

    fig.add_trace(go.Scatter(
        x=k_theory_plot,
        y=p_theory,
        mode='lines',
        line=dict(color=ACCENT_COLOR, width=2.5),
        name='Poisson',
        hoverinfo='skip'
    ), row=1, col=2)

    # Add frames to figure
    fig.frames = frames

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Random Network Growth</b><br><span style='font-size:14px;color:#888'>Erdős-Rényi G(n,p) Model</span>",
            x=0.5,
            y=0.95,
            font=dict(size=22, color=TEXT_COLOR, family="Arial Black, sans-serif")
        ),
        showlegend=True,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='rgba(100,100,120,0.5)',
            borderwidth=1,
            font=dict(size=12, color=TEXT_COLOR)
        ),
        hovermode='closest',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        width=1300,
        height=700,
        margin=dict(l=40, r=40, t=120, b=120),
        font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=14, color=TEXT_COLOR),
                prefix="Step: ",
                visible=True,
                xanchor="center"
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=40),
            len=0.92,
            x=0.04,
            y=0.08,
            bgcolor='#2a2a4a',
            bordercolor='#3a3a5a',
            borderwidth=1,
            ticklen=4,
            font=dict(color=TEXT_COLOR),
            steps=slider_steps
        )],
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                direction="right",
                y=0.08,
                x=0.04,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=0, r=10, b=60),
                bgcolor='#2a2a4a',
                bordercolor='#3a3a5a',
                font=dict(color=TEXT_COLOR, size=11),
                buttons=[
                    dict(
                        label=" Pause ",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label=" 1x ",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=80, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label=" 2x ",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=40, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label=" 4x ",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=20, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        annotations=[
            dict(
                text="<b>Network Graph</b>",
                xref="paper", yref="paper",
                x=0.26, y=1.08,
                showarrow=False,
                font=dict(size=16, color=TEXT_COLOR)
            ),
            dict(
                text="<b>Degree Distribution</b>",
                xref="paper", yref="paper",
                x=0.82, y=1.08,
                showarrow=False,
                font=dict(size=16, color=TEXT_COLOR)
            ),
            dict(
                text=f"<b>N = {len(nodes)}</b>  |  <b>E = {len(edges)}</b>  |  <b>p = {p}</b>",
                xref="paper", yref="paper",
                x=0.26, y=1.02,
                showarrow=False,
                font=dict(size=14, color='#888')
            ),
            dict(
                text="<i>Drag slider or use speed controls to animate</i>",
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                showarrow=False,
                font=dict(size=11, color='#666')
            )
        ]
    )

    # Configure network plot axes
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        row=1, col=1
    )

    # Configure degree distribution axes (linear scale for Poisson)
    fig.update_xaxes(
        title=dict(
            text="Degree (k)",
            font=dict(size=14, color=TEXT_COLOR),
            standoff=10
        ),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showline=True,
        linecolor=GRID_COLOR,
        linewidth=1,
        tickfont=dict(size=11, color=TEXT_COLOR),
        row=1, col=2
    )
    fig.update_yaxes(
        title=dict(
            text="P(k)",
            font=dict(size=14, color=TEXT_COLOR),
            standoff=10
        ),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showline=True,
        linecolor=GRID_COLOR,
        linewidth=1,
        tickfont=dict(size=11, color=TEXT_COLOR),
        row=1, col=2
    )

    # Save to HTML
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Visualization saved to {output_file}")

    return fig


def main():
    from pathlib import Path

    # Output to results folder
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "er_network.html"

    # Parameters
    n_final = 500
    p = 0.008  # Edge probability (gives similar avg degree to BA with m=2)

    print(f"Generating ER network history ({n_final} steps, p={p})...")
    history = generate_er_network_history(n_final=n_final, p=p, seed=42)

    print("Computing stable layout for all nodes...")
    pos = compute_layout_for_all_nodes(n_final, history, seed=42)

    print("Creating interactive visualization...")
    create_visualization(history, pos, p=p, output_file=str(output_file))

    # Print statistics about the final network
    final_nodes, final_edges = history[-1]
    G = nx.Graph()
    G.add_nodes_from(final_nodes)
    G.add_edges_from(final_edges)

    degrees = [d for n, d in G.degree()]
    expected_avg_degree = (n_final - 1) * p

    print(f"\nFinal network statistics:")
    print(f"  Nodes: {len(final_nodes)}")
    print(f"  Edges: {len(final_edges)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Avg degree: {sum(degrees)/len(degrees):.2f} (expected: {expected_avg_degree:.2f})")
    print(f"  Edge probability p: {p}")
    print(f"\nOpen {output_file} in your browser to view the visualization.")


if __name__ == "__main__":
    main()
