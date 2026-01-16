"""
Scale-Free Network Visualization using Barabási-Albert Preferential Attachment

Generates an interactive HTML visualization showing network growth from 0 to 200 nodes.
"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random

def generate_ba_network_history(n_final=200, m=2, seed=42):
    """
    Generate BA network growth history step by step.

    Args:
        n_final: Final number of nodes
        m: Number of edges each new node creates
        seed: Random seed for reproducibility

    Returns:
        List of (nodes, edges) tuples for each step
    """
    random.seed(seed)
    np.random.seed(seed)

    history = []

    # Start with initial complete graph of m nodes
    G = nx.complete_graph(m)
    history.append((list(G.nodes()), list(G.edges())))

    # Track degree for preferential attachment
    degrees = {i: m - 1 for i in range(m)}  # Each node in complete graph has m-1 edges

    for new_node in range(m, n_final):
        # Calculate attachment probabilities based on degree
        total_degree = sum(degrees.values())
        nodes = list(degrees.keys())
        probs = [degrees[n] / total_degree for n in nodes]

        # Select m unique nodes to connect to (preferential attachment)
        targets = set()
        while len(targets) < m:
            chosen = np.random.choice(nodes, p=probs)
            targets.add(chosen)

        # Add new node and edges
        G.add_node(new_node)
        degrees[new_node] = 0

        for target in targets:
            G.add_edge(new_node, target)
            degrees[new_node] += 1
            degrees[target] += 1

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
    This is the standard approach for visualizing power-law distributions.

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
    # Use base 2 for cleaner bin boundaries
    log_min = np.log10(min_deg)
    log_max = np.log10(max_deg * 1.1)  # Slightly extend to include max
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
            # Geometric mean for bin center (appropriate for log-scale)
            bin_center = np.sqrt(left_edge * right_edge)

            # Linear bin width for probability density
            bin_width = right_edge - left_edge

            # Probability density: P(k) = (count/N) / dk
            # This ensures integral of P(k)dk = 1
            prob_density = count / (n_total * bin_width)

            bin_centers.append(bin_center)
            probabilities.append(prob_density)

    return bin_centers, probabilities


def create_visualization(history, pos, output_file="ba_network.html", frame_step=5):
    """
    Create interactive Plotly visualization with slider and degree distribution plot.
    Professional dark theme design inspired by Veritasium.

    Args:
        history: List of (nodes, edges) tuples
        pos: Node positions dict
        output_file: Output HTML filename
        frame_step: Sample every nth step to reduce file size
    """
    # Sample frames to reduce file size
    sampled_indices = list(range(0, len(history), frame_step))
    if sampled_indices[-1] != len(history) - 1:
        sampled_indices.append(len(history) - 1)  # Always include final state

    n_steps = len(sampled_indices)

    # Precompute node degrees for each step
    degrees_history = []
    for nodes, edges in history:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        degrees_history.append(dict(G.degree()))

    # Color scheme - professional dark theme
    BG_COLOR = '#1a1a2e'
    PLOT_BG = '#16213e'
    TEXT_COLOR = '#eaeaea'
    GRID_COLOR = '#2a2a4a'
    EDGE_COLOR = 'rgba(100, 140, 180, 0.4)'
    ACCENT_COLOR = '#e94560'
    DATA_COLOR = '#4cc9f0'

    # Create subplots with better spacing
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.12,
        subplot_titles=("", "")  # We'll add custom titles
    )

    # Create frames for sampled steps
    frames = []
    slider_steps = []

    # Get max degree for consistent colorbar scaling
    max_degree_overall = max(max(degrees_history[i].values()) for i in sampled_indices)

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
                colorscale='Plasma',
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

        # Degree distribution trace (log-binned)
        bin_centers, probabilities = compute_logbinned_degree_distribution(degree_list)

        # Add theoretical power-law line P(k) ~ k^(-3) for BA networks
        if len(bin_centers) >= 2:
            k_theory = np.logspace(np.log10(min(bin_centers)), np.log10(max(bin_centers) * 1.5), 50)
            p_theory = k_theory ** (-3)
            mid_idx = len(bin_centers) // 2
            scale = probabilities[mid_idx] / (bin_centers[mid_idx] ** (-3))
            p_theory = scale * k_theory ** (-3)
        else:
            k_theory = [1, 10]
            p_theory = [1, 0.001]

        degree_trace = go.Scatter(
            x=bin_centers if bin_centers else [1],
            y=probabilities if probabilities else [1],
            mode='markers',
            marker=dict(
                size=10,
                color=DATA_COLOR,
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name='Empirical',
            hovertemplate='k = %{x:.1f}<br>P(k) = %{y:.2e}<extra></extra>',
            xaxis='x2', yaxis='y2'
        )

        # Theoretical line trace
        theory_trace = go.Scatter(
            x=k_theory,
            y=p_theory,
            mode='lines',
            line=dict(color=ACCENT_COLOR, width=2.5),
            name='P(k) ~ k⁻³',
            hoverinfo='skip',
            xaxis='x2', yaxis='y2'
        )

        frame = go.Frame(
            data=[edge_trace, node_trace, degree_trace, theory_trace],
            name=str(frame_idx),
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"<b>N = {len(nodes)}</b>  |  <b>E = {len(edges)}</b>",
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

        # Slider step - show fewer labels for cleaner look
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
            colorscale='Plasma',
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
    bin_centers, probabilities = compute_logbinned_degree_distribution(degree_list)

    fig.add_trace(go.Scatter(
        x=bin_centers if bin_centers else [1],
        y=probabilities if probabilities else [1],
        mode='markers',
        marker=dict(
            size=10,
            color=DATA_COLOR,
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        name='Empirical',
        hovertemplate='k = %{x:.1f}<br>P(k) = %{y:.2e}<extra></extra>'
    ), row=1, col=2)

    # Add theoretical line
    if len(bin_centers) >= 2:
        k_theory = np.logspace(np.log10(min(bin_centers)), np.log10(max(bin_centers) * 1.5), 50)
        mid_idx = len(bin_centers) // 2
        scale = probabilities[mid_idx] / (bin_centers[mid_idx] ** (-3))
        p_theory = scale * k_theory ** (-3)
    else:
        k_theory = [1, 10]
        p_theory = [1, 0.001]

    fig.add_trace(go.Scatter(
        x=k_theory,
        y=p_theory,
        mode='lines',
        line=dict(color=ACCENT_COLOR, width=2.5),
        name='P(k) ~ k⁻³',
        hoverinfo='skip'
    ), row=1, col=2)

    # Add frames to figure
    fig.frames = frames

    # Update layout - professional dark theme
    fig.update_layout(
        title=dict(
            text="<b>Scale-Free Network Growth</b><br><span style='font-size:14px;color:#888'>Barabási-Albert Preferential Attachment Model</span>",
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
        # Custom annotations for subplot titles and stats
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
                text=f"<b>N = {len(nodes)}</b>  |  <b>E = {len(edges)}</b>",
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

    # Configure network plot axes (clean, no ticks)
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

    # Configure degree distribution axes (log-log scale, professional styling)
    fig.update_xaxes(
        type="log",
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
        minor=dict(showgrid=True, gridcolor='rgba(50,50,80,0.5)'),
        row=1, col=2
    )
    fig.update_yaxes(
        type="log",
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
        minor=dict(showgrid=True, gridcolor='rgba(50,50,80,0.5)'),
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
    output_file = output_dir / "ba_network.html"

    print("Generating BA network history (500 steps)...")
    history = generate_ba_network_history(n_final=500, m=2, seed=42)

    print("Computing stable layout for all nodes...")
    pos = compute_layout_for_all_nodes(500, history, seed=42)

    print("Creating interactive visualization...")
    create_visualization(history, pos, output_file=str(output_file))

    # Print some statistics about the final network
    final_nodes, final_edges = history[-1]
    G = nx.Graph()
    G.add_nodes_from(final_nodes)
    G.add_edges_from(final_edges)

    degrees = [d for n, d in G.degree()]
    print(f"\nFinal network statistics:")
    print(f"  Nodes: {len(final_nodes)}")
    print(f"  Edges: {len(final_edges)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Avg degree: {sum(degrees)/len(degrees):.2f}")
    print(f"\nOpen {output_file} in your browser to view the visualization.")


if __name__ == "__main__":
    main()
