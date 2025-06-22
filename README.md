Bachelor’s thesis in Statistics on the topic of random graphs and their properties with title "Random Graph Generator with Degree Constraints and Hub Nodes".

This project generates, visualizes, analyzes, and saves undirected random graphs with controllable maximum node degree and a definable proportion of hub nodes. It is useful for graph theory experiments, complex network analysis, and simulations of structural graph properties.

## Main Features

- **Random graph generation** with binomial edge distribution (Erdős-Rényi model).
- **Control of maximum node degree**, allowing a proportion of nodes to act as hubs exempt from the degree constraint.
- **Degree reduction** to enforce degree constraints, using a randomized approach.
- **Visualization**:
  - Circular layout of nodes.
  - Compact bipartite graph layout.
  - Coloring connected components (clusters).
  - Histogram of degree distribution.
- **Structural property analysis** of generated graphs:
  - Connectivity
  - Bipartiteness
  - Acyclicity
- **Statistical study** of graph properties by varying parameters (n, p).
- **Saving and loading** graphs to/from files in edge list format.

## Project Structure

- `random_graph_with_hub_nodes(...)`: Generates a random graph with degree constraints.
- `degree_reduction(...)`: Reduces node degrees to respect maximum degree limits.
- `choose_nodes_to_adjust(...)`: Selects nodes exceeding the max degree excluding hubs.
- `draw_circular_graph(...)`: Simple circular graph visualization.
- `draw_subgraphs(...)`: Visualizes connected components with distinct colors.
- `histogram_of_graph_degrees(...)`: Degree distribution histogram.
- `compare_degree_distributions(...)`: Visual comparison of degree distributions for different p values.
- `plot_3d_property_frequency(...)`: 3D plot showing frequency of structural properties over n and p.
- `write_graphs_on_file(...)`: Saves generated graphs to a file.
- `read_graphs_from_file(...)`: Reads saved graphs from a file.

## Main Parameters

- `n`: Number of nodes in the graph.
- `p`: Probability of edge creation (Erdős-Rényi model).
- `max_degree`: Maximum allowed degree for non-hub nodes.
- `ratio_hub_nodes`: Proportion of nodes exempt from the max degree constraint (hubs).

Example usage:
```python
# Generate graph with max degree constraint
g = random_graph_with_hub_nodes(30, 0.2, max_degree=5, ratio_hub_nodes=0.1, print_details=True)
draw_circular_graph(g)
histogram_of_graph_degrees(g)
