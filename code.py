import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import time
import pprint
import tkinter as tk
from tkinter import ttk
from matplotlib.cm import Blues

# Suggested input values for section 4:
# n = 20, 30, 40, 50
# p = 0.5, 0.1, 0.15, 0.2, 0.25
# or
# n = 15, 20, 25, 30, 35
# p = 0.01, 0.03, 0.05, 0.07, 0.09

# --------------------------- Graph generation -----------------------------

# Select nodes randomly to reduce their degree
# d = array of node degrees
# ratio_hub_nodes = proportion of nodes allowed to have arbitrary degree
# max_degree = maximum allowed degree for other nodes
def choose_nodes_to_adjust(n, d, ratio_hub_nodes, max_degree):
    n_hub_nodes = int(n * ratio_hub_nodes)  # number of nodes allowed to exceed max_degree
    is_hub_node = []  # nodes currently exceeding max_degree

    for i in range(n):
        if d[i] > max_degree:
            is_hub_node.append(i)
    
    if len(is_hub_node) <= n_hub_nodes:
        nodes_to_adjust = []  # degrees are within limits; no adjustment needed
    else:
        # Randomly select which nodes to keep as hubs; others must reduce degree
        choose_hub_nodes = np.random.choice(is_hub_node, n_hub_nodes, replace=False)
        nodes_to_adjust = np.setdiff1d(is_hub_node, choose_hub_nodes)

    return nodes_to_adjust

# Reduce degrees of specified nodes by removing edges
# m = adjacency matrix
# g = graph object
# max_degree = target max degree
# index = list of node indices to adjust
def degree_reduction(m, g, max_degree, index):
    d = m.sum(axis=1)
    n = len(d)
    for i in index:
        while d[i] > max_degree:
            old_edges = [j for j in range(n) if m[i][j] == 1]  # active edges of node i
            to_be_off = int(d[i] - max_degree)  # number of edges to remove
            off = np.random.choice(old_edges, to_be_off, replace=False)  # edges to remove

            for k in off:
                m[i][k] = 0
                m[k][i] = 0  # symmetry in adjacency matrix
                if i < k:
                    g.remove_edge(i, k)
                else:
                    g.remove_edge(k, i)
            d = m.sum(axis=1)  # update degrees after removals

    return g, m  # return updated graph and adjacency matrix

# Generate a random undirected graph with controlled degree distribution
# n = number of nodes
# p = edge creation probability
# max_degree = maximum allowed node degree (optional)
# ratio_hub_nodes = proportion of nodes allowed to exceed max_degree (default 0)
# print_details = flag to print graph stats
def random_graph_with_hub_nodes(n, p, max_degree=None, ratio_hub_nodes=0, print_details=False):
    
    # Input validation
    if p < 0 or p > 1:
        print("Error: p must be in [0,1].\n")
        return None
    if n < 0 or (n % 1 != 0):
        print("Error: n must be a positive integer.\n")
        return None
    if ratio_hub_nodes < 0 or ratio_hub_nodes > 1:
        print("Error: ratio_hub_nodes must be in [0,1].\n")
        return None
    if ratio_hub_nodes != 0 and max_degree is None:
        print("Error: max_degree must be set if ratio_hub_nodes > 0.\n")
        return None
    if max_degree is not None:
        if max_degree < 0 or max_degree > n or (max_degree % 1 != 0):
            print("Error: max_degree must be an integer in [0, n].\n")
            return None

    n_edges_possible = int((n**2 - n) / 2)  # number of possible edges in undirected graph without loops
    edges = [np.random.binomial(1, p) for _ in range(n_edges_possible)]
    adjacency_matrix = np.zeros((n, n), dtype=int)
    Graph = nx.Graph()
    Graph.add_nodes_from(range(n))

    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edges[index] == 1:
                adjacency_matrix[i][j] = 1
                Graph.add_edge(i, j)
            index += 1

    adjacency_matrix += adjacency_matrix.T  # make adjacency matrix symmetric

    if max_degree is not None:
        before = adjacency_matrix.sum(axis=1)
        after = before.copy()

        pawns = []  # nodes to adjust

        if 0 < ratio_hub_nodes < 1:
            pawns = choose_nodes_to_adjust(n, before, ratio_hub_nodes, max_degree)
        elif ratio_hub_nodes == 0:
            pawns = [i for i in range(n) if before[i] > max_degree]

        if pawns:
            Graph, adjacency_matrix = degree_reduction(adjacency_matrix, Graph, max_degree, pawns)
            after = adjacency_matrix.sum(axis=1)

    if print_details:
        n_edges = Graph.number_of_edges()
        density = round(nx.density(Graph), 3)
        if max_degree is not None:
            print(f"\nDegrees before: {before}\nDegrees after: {after}\nEdges removed: {int(sum(before - after) / 2)}\n")
        else:
            print(f"\nDegrees: {adjacency_matrix.sum(axis=1)}\n")
        print(f"Edges: {n_edges} out of possible {n*(n-1)//2}\nDensity: {density}\nAdjacency matrix:\n{adjacency_matrix}\nNodes: {list(Graph.nodes)}\nEdges:")
        pprint.pprint(list(Graph.edges))

    return Graph

# ------------------------- Graph visualization ----------------------------

def draw_circular_graph(Graph):
    pos = nx.circular_layout(Graph)
    nx.draw_networkx_nodes(Graph, pos, node_size=500, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(Graph, pos, font_size=10, font_color='black')
    nx.draw_networkx_edges(Graph, pos, width=1.2, edge_color='black')
    plt.axis('off')
    return plt.show()

def draw_compact_bipartite_graph(G):
    pos = nx.bipartite_layout(G, [node for node in G.nodes if G.nodes[node]['bipartite'] == 0])
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='black')
    plt.title("Undirected probabilistic graph")
    plt.axis('off')
    plt.show()

# Visualize degree distribution histogram
def histogram_of_graph_degrees(Graph):
    n_nodes = nx.number_of_nodes(Graph)
    hist = nx.degree_histogram(Graph)
    hist = [x / n_nodes for x in hist]

    # Extend histogram to full range of degrees
    while len(hist) < n_nodes:
        hist.append(0)

    plt.bar(range(n_nodes), hist, align='center')
    plt.ylim(0, max(hist) + 0.01)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree distribution histogram')
    return plt.show()

# Visualize connected components with distinct colors
def draw_subgraphs(Graph):
    components = list(nx.connected_components(Graph))
    col_dict = {}
    node_colors = []

    colors = [Blues(0.25 + (i / len(components)) * 0.75) for i in range(len(components))]
    # Restrict color range to avoid near-white shades

    for i, comp in enumerate(components):
        for node in comp:
            col_dict[node] = colors[i]

    pos = nx.spring_layout(Graph)  # force-directed layout to highlight clusters

    for node in Graph.nodes():
        node_colors.append(col_dict[node])

    nx.draw(Graph, pos, node_color=node_colors, with_labels=True)
    plt.show()

# Comparison operations -----------------------------------------------------------------

# Plot for comparing degree distributions for varying p values:
def compare_degree_distributions(n, p_values, max_degree=None, ratio_hub_nodes=0):
    _, axs = plt.subplots(len(p_values), 1)  # Create subplots with one row per p value

    plt.suptitle('Degree Distributions')  # Main title for the figure

    max_y = 0  # Track maximum y-axis value for consistent scale across subplots
    max_x = 0  # Track maximum x-axis value

    for i in range(len(p_values)):
        graph = random_graph_with_hub_nodes(n, p_values[i], max_degree, ratio_hub_nodes)

        n_nodes = nx.number_of_nodes(graph)
        hist = nx.degree_histogram(graph)  # Absolute degree frequencies
        hist = [x / n_nodes for x in hist]  # Convert to relative frequencies        
        
        # Find axis limits to include max range in all histograms
        max_x = max(max_x, len(hist))
        max_y = max(max_y, max(hist))

        while len(hist) < n_nodes:
            hist.append(0)  # Pad degree axis for consistent x-axis length

        axs[i].bar(range(n_nodes), hist, align='center')
        axs[i].set_title(f'n={n} p={p_values[i]} max_degree={max_degree} ratio_hub_nodes={ratio_hub_nodes}', fontsize=8)

    for ax in axs:
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y + 0.01)
    
    plt.tight_layout()
    return plt.show()


# Property check functions:
def connectivity(Graph):
    return nx.is_connected(Graph)

def bipartition(Graph):
    return nx.is_bipartite(Graph)

def aciclicity(Graph):
    return nx.is_forest(Graph)

# Function to calculate the proportion of graphs with certain properties (connected, bipartite, acyclic)
# for a batch of graphs generated with different n and p values
# mc = number of samples per combination of n, p
# property = property function to check
def plot_3d_property_frequency(n_values, p_values, mc=40, max_degree=None, ratio_hub_nodes=0, property='', title='Frequency'):
    
    if property not in [connectivity, bipartition, aciclicity]:
        print('Error: property function must be one of [connectivity, bipartition, aciclicity].')
        return None

    n_values = sorted(n_values)  # Sort n values
    p_values = sorted(p_values)  # Sort p values

    freq = np.zeros((len(n_values), len(p_values)))  # Frequencies of graphs satisfying the property
    one_sample = [0] * mc  # Store property results for sample graphs for each n, p combination

    for i in range(len(n_values)):
        print(n_values[i], '---------------------')  # Progress indicator
        for j in range(len(p_values)):
            for k in range(mc):
                g = random_graph_with_hub_nodes(n=n_values[i], p=p_values[j], max_degree=max_degree, ratio_hub_nodes=ratio_hub_nodes)
                one_sample[k] = property(g)  # 1 if graph satisfies property, else 0

            freq[i][j] = np.mean(one_sample)  # Proportion of graphs satisfying the property

    x, y = np.meshgrid(p_values, n_values)
    z = freq

    fig = plt.figure()
    gr = fig.add_subplot(111, projection='3d')
    gr.plot_surface(x, y, z, cmap='viridis')  # Surface plot with 'viridis' colormap

    gr.set_xlabel('p')
    gr.set_ylabel('n')
    gr.set_zlabel('Frequency')
    gr.set_title(title)

    return plt.show()


# File saving and loading ---------------------------------------------------------

# Function to save graphs to a file by writing their edges for samples with varying n, p
# file_name = name of the output file
def write_graphs_on_file(n_list, p_list, max_degree=None, ratio_hub_nodes=0, mc=40, file_name='graphs.txt'):
    start = time.time()

    n_list = sorted(n_list)
    p_list = sorted(p_list)

    with open(file_name, 'wb') as file:
        for n in n_list:
            print(n, '---------------------------')

            md = max_degree if max_degree is not None else n - 1

            for p in p_list:
                n = int(n)
                p = round(p, 2)
                file.write(f'{n} {p} {md} {ratio_hub_nodes} {mc}\n'.encode())
                for k in range(mc):  
                    file.write(f'{k}\n'.encode())
                    g = random_graph_with_hub_nodes(n, p, max_degree, ratio_hub_nodes)
                    nx.write_edgelist(g, file, data=False) 

    stop = time.time()
    print('Run time:', (stop - start), 'seconds')
    return file_name

def read_graphs_from_file(file_name):
    try:
        with open(file_name, 'rb') as file:
            graphs_dict = {}  # Dictionary to hold lists of graphs keyed by (n, p, max_degree, ratio_hub_nodes, mc)
            graph_list = []   # List of graphs for the current parameter set
            g = None          # Current graph being read
            n = p = md = rhn = mc = None  # Parameters for the current set of graphs

            for line in file:
                line = line.decode().strip()  # Decode bytes to string and strip whitespace
                parts = line.split()           # Split line into tokens

                if len(parts) == 5:
                    # New graph parameter set header line encountered
                    # Save previous graph and list if present
                    if g is not None:
                        graph_list.append(g)
                        graphs_dict[(n, p, md, rhn, mc)] = graph_list

                    n, p, md, rhn, mc = map(float, parts)  # Read parameters
                    graph_list = []
                    g = None

                else:
                    if len(parts) == 1:
                        # Start of a new individual graph in current set
                        if g is not None:
                            graph_list.append(g)  # Save previous graph

                        g = nx.Graph()
                        for i in range(int(n)):
                            g.add_node(i)  # Add n nodes numbered 0 to n-1

                    elif len(parts) == 2:
                        # Edge line: add edge to current graph
                        u, v = map(int, parts)
                        g.add_edge(u, v)

            # After loop ends, save the last graph list if parameters were set
            if n is not None and p is not None and g is not None:
                graph_list.append(g)
                graphs_dict[(n, p, md, rhn, mc)] = graph_list

        return graphs_dict

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None

import tkinter as tk
from tkinter import ttk
import networkx as nx

# ----------------------------------------
# User Interface Functions
# ----------------------------------------

def section_single_graph():
    # Retrieve parameters from inputs
    n = int(n_par1.get())
    p = float(p_par1.get())
    
    max_degree_str = md_par1.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par1.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    details = bool(det_par_var1.get())

    # Generate graph using user parameters
    g = random_graph_with_hub_nodes(n, p, max_degree, ratio_hub_nodes, details)
    return g

def section_single_graph_property():
    g = section_single_graph()
    
    # Display properties of the graph
    result_label1.config(text=(
        f"Connectivity: {connectivity(g)}\n"
        f"Acyclicity: {aciclicity(g)}\n"
        f"Bipartite: {bipartition(g)}"
    ))
    
    # Get selected graph display option
    option = menu_onegraph1.get()
    
    if option == "Grafico circolare":
        draw_circular_graph(g)
    elif option == "Rappresentazione dei sottografi":
        draw_subgraphs(g)
    elif option == "Istogramma dei gradi":
        histogram_of_graph_degrees(g)

def section_writing_on_file():
    # Parse comma-separated input lists for n and p
    n_list = [int(x.strip()) for x in n_val2.get().split(',')]
    p_list = [float(x.strip()) for x in p_val2.get().split(',')]
    
    max_degree_str = md_par2.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par2.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0
    
    mc = int(mc_par2.get())
    file_write = fw_str2.get().strip()
    
    write_graphs_on_file(n_list, p_list, max_degree, ratio_hub_nodes, mc, file_write)

def section_read_file1():
    file_read = fr_str3.get().strip()
    graphs_dict = read_graphs_from_file(file_read)
    
    if graphs_dict is None:
        result_label3.config(text=f"The file '{file_read}' does not exist.")
        return None
    
    # Collect unique n and p values from keys
    n_values, p_values = [], []
    for key in graphs_dict.keys():
        n, p, md, rhn, mc = key
        if n not in n_values:
            n_values.append(int(n))
        if p not in p_values:
            p_values.append(p)
    
    # Display parameters summary
    md_display = None if md == (max(n_values) - 1) else int(md)
    result_label3.config(text=(
        f"The file contains graphs generated with parameters:\n"
        f"n: {n_values}\n"
        f"p: {p_values}\n"
        f"max_degree: {md_display}\n"
        f"ratio_hub_nodes: {rhn}\n"
        f"mc: {int(mc)}"
    ))
    return graphs_dict

def section_read_file2():
    graphs_dict = section_read_file1()
    if graphs_dict is None:
        return
    
    # Retrieve parameters and keys
    all_keys = list(graphs_dict.keys())
    n_values, p_values = [], []
    for key in all_keys:
        n, p, md, rhn, mc = key
        if n not in n_values:
            n_values.append(int(n))
        if p not in p_values:
            p_values.append(p)

    # Get selected parameters to choose graph
    selected_params = [float(x.strip()) for x in par_str3.get().split(',')]
    
    # Find the matching key with selected n and p
    selected_key = next((k for k in all_keys if k[:2] == tuple(selected_params)), None)
    if selected_key is None:
        result_label33.config(text="No graph found for selected parameters.")
        return

    # Select graph by index
    chosen_mc = int(mc_num3.get())
    g = graphs_dict[selected_key][chosen_mc]

    # Show graph properties
    result_label33.config(text=(
        f"Connectivity: {connectivity(g)}\n"
        f"Acyclicity: {aciclicity(g)}\n"
        f"Bipartite: {bipartition(g)}"
    ))

    # Display graph based on selection
    option = menu_drawgraph3.get()
    if option == "Grafico circolare":
        draw_circular_graph(g)
    elif option == "Rappresentazione dei sottografi":
        draw_subgraphs(g)
    elif option == "Istogramma dei gradi":
        histogram_of_graph_degrees(g)

def section_histogram_of_freq():
    n_list = [int(x.strip()) for x in n_val4.get().split(',')]
    p_list = [float(x.strip()) for x in p_val4.get().split(',')]
    
    max_degree_str = md_par4.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par4.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    mc = int(mc_par4.get())
    
    property_choice = menu_property4.get()
    title_map = {
        "Connettività": "Frequenza di grafi connessi",
        "Aciclicità": "Frequenza di grafi aciclici",
        "Bipartizione": "Frequenza di grafi bipartiti"
    }

    property_func_map = {
        "Connettività": connectivity,
        "Aciclicità": aciclicity,
        "Bipartizione": bipartition
    }

    if property_choice in property_func_map:
        plot_3d_property_frequency(
            n_list, p_list, mc, max_degree, ratio_hub_nodes,
            property=property_func_map[property_choice],
            title=title_map[property_choice]
        )

def section_compare_degree_distribution():
    n = int(n_par5.get())
    p_list = [float(x.strip()) for x in p_val5.get().split(',')]
    
    max_degree_str = md_par5.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par5.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    compare_degree_distributions(n, p_list, max_degree, ratio_hub_nodes)

# ----------------------------------------
# Tkinter GUI Setup
# ----------------------------------------

app = tk.Tk()
app.title("User Interface")

notebook = ttk.Notebook(app)
notebook.pack(fill='both', expand=True)

# Section 1: Single Graph ---------------------------------
section1 = ttk.Frame(notebook)
notebook.add(section1, text='Single Graph')

ttk.Label(section1, text="Number of nodes:").pack()
n_par1 = ttk.Entry(section1)
n_par1.pack()

ttk.Label(section1, text="Connection probability:").pack()
p_par1 = ttk.Entry(section1)
p_par1.pack()

ttk.Label(section1, text="Maximum degree:").pack()
md_par1 = ttk.Entry(section1)
md_par1.pack()

ttk.Label(section1, text="Ratio of hub nodes:").pack()
rhn_par1 = ttk.Entry(section1)
rhn_par1.pack()

det_par_var1 = tk.IntVar()
ttk.Label(section1, text="Show details:").pack()
ttk.Checkbutton(section1, variable=det_par_var1).pack()

ttk.Label(section1, text="Graph type:").pack()
menu_onegraph1 = ttk.Combobox(section1, values=[
    "Grafico circolare",
    "Rappresentazione dei sottografi",
    "Istogramma dei gradi"
])
menu_onegraph1.pack()

ttk.Button(section1, text="Execute", command=section_single_graph_property).pack()
result_label1 = ttk.Label(section1, text="")
result_label1.pack()

# Section 2: Write Graphs to File ---------------------------
section2 = ttk.Frame(notebook)
notebook.add(section2, text='Write to File')

ttk.Label(section2, text="List of n values (comma separated):").pack()
n_val2 = ttk.Entry(section2)
n_val2.pack()

ttk.Label(section2, text="List of p values (comma separated):").pack()
p_val2 = ttk.Entry(section2)
p_val2.pack()

ttk.Label(section2, text="Maximum degree (optional):").pack()
md_par2 = ttk.Entry(section2)
md_par2.pack()

ttk.Label(section2, text="Ratio of hub nodes (default 0):").pack()
rhn_par2 = ttk.Entry(section2)
rhn_par2.pack()

ttk.Label(section2, text="Number of replications:").pack()
mc_par2 = ttk.Entry(section2)
mc_par2.pack()

ttk.Label(section2, text="File name:").pack()
fw_str2 = ttk.Entry(section2)
fw_str2.pack()

ttk.Button(section2, text="Execute", command=section_writing_on_file).pack()
risultato_label2 = ttk.Label(section2, text="")
risultato_label2.pack()

# Section 3: Read from File ---------------------------------
section3 = ttk.Frame(notebook)
notebook.add(section3, text='Read from File')

ttk.Label(section3, text="File name:").pack()
fr_str3 = ttk.Entry(section3)
fr_str3.pack()

ttk.Button(section3, text="Load", command=section_read_file1).pack()
result_label3 = ttk.Label(section3, text="")
result_label3.pack()

ttk.Label(section3, text="Select parameters (n,p):").pack()
par_str3 = ttk.Entry(section3)
par_str3.pack()

ttk.Label(section3, text="Element index:").pack()
mc_num3 = ttk.Entry(section3)
mc_num3.pack()

ttk.Label(section3, text="Graph type:").pack()
menu_drawgraph3 = ttk.Combobox(section3, values=[
    "Grafico circolare",
    "Rappresentazione dei sottografi",
    "Istogramma dei gradi"
])
menu_drawgraph3.pack()

ttk.Button(section3, text="Show Graph", command=section_read_file2).pack()
result_label33 = ttk.Label(section3, text="")
result_label33.pack()

# Section 4: 3D Plot -----------------------------------------
section4 = ttk.Frame(notebook)
notebook.add(section4, text='3D Plot')

ttk.Label(section4, text="List of n values (comma separated):").pack()
n_val4 = ttk.Entry(section4)
n_val4.pack()

ttk.Label(section4, text="List of p values (comma separated):").pack()
p_val4 = ttk.Entry(section4)
p_val4.pack()

ttk.Label(section4, text="Maximum degree:").pack()
md_par4 = ttk.Entry(section4)
md_par4.pack()

ttk.Label(section4, text="Ratio of hub nodes:").pack()
rhn_par4 = ttk.Entry(section4)
rhn_par4.pack()

ttk.Label(section4, text="Number of samples:").pack()
mc_par4 = ttk.Entry(section4)
mc_par4.pack()

ttk.Label(section4, text="Property to check:").pack()
menu_property4 = ttk.Combobox(section4, values=[
    "Connettività",
    "Aciclicità",
    "Bipartizione"
])
menu_property4.pack()

ttk.Button(section4, text="Execute", command=section_histogram_of_freq).pack()

# Section 5: Compare Degree Distributions -------------------
section5 = ttk.Frame(notebook)
notebook.add(section5, text='Degree Distribution')

ttk.Label(section5, text="Number of nodes:").pack()
n_par5 = ttk.Entry(section5)
n_par5.pack()

ttk.Label(section5, text="List of p values (comma separated):").pack()
p_val5 = ttk.Entry(section5)
p_val5.pack()

ttk.Label(section5, text="Maximum degree:").pack()
md_par5 = ttk.Entry(section5)
md_par5.pack()

ttk.Label(section5, text="Ratio of hub nodes:").pack()
rhn_par5 = ttk.Entry(section5)
rhn_par5.pack()

ttk.Button(section5, text="Execute", command=section_compare_degree_distribution).pack()

# ----------------------------------------
# Start the GUI event loop
# ----------------------------------------
app.mainloop()
