import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import time
import pprint
import tkinter as tk
from tkinter import ttk
from matplotlib.cm import Blues

# Esempi consigliati per test:
# n = 20,30,40,50 | p = 0.5,0.1,0.15,0.2,0.25
# n = 15,20,25,30,35 | p = 0.01,0.03,0.05,0.07,0.09

# Seleziona i nodi con grado superiore al massimo consentito da modificare
def choose_nodes_to_adjust(n, d, ratio_hub_nodes, max_degree):
    n_hub_nodes = int(n * ratio_hub_nodes)  # Numero di nodi ammessi con grado arbitrario
    is_hub_node = []

    # Individuare i nodi con grado superiore al massimo
    for i in range(n):
        if d[i] > max_degree:
            is_hub_node.append(i)
    
    # Nessun intervento necessario se i nodi fuori soglia rientrano nel limite consentito
    if len(is_hub_node) <= n_hub_nodes:
        nodes_to_adjust = []
    else:
        # Estrazione casuale dei nodi da mantenere come hub
        choose_hub_nodes = np.random.choice(is_hub_node, n_hub_nodes, replace=False)
        nodes_to_adjust = np.setdiff1d(is_hub_node, choose_hub_nodes)
        
    return nodes_to_adjust

# Riduzione del grado dei nodi selezionati tramite rimozione di archi
def degree_reduction(m, g, max_degree, index):
    d = m.sum(axis=1)
    n = len(d)

    for i in index:
        # Ridurre il grado fino a rientrare nel valore massimo consentito
        while d[i] > max_degree:
            old_edges = []
            to_be_off = int(d[i] - max_degree)

            # Identificare i nodi connessi al nodo corrente
            for j in range(n):
                if m[i][j] == 1:
                    old_edges.append(j)
            
            # Selezionare casualmente gli archi da rimuovere
            off = np.random.choice(old_edges, to_be_off, replace=False)

            for k in off:
                m[i][k] = 0
                m[k][i] = 0  # La matrice di adiacenza deve rimanere simmetrica (grafo non orientato)

                # Rimuovere l’arco dal grafo
                if i < k:
                    g.remove_edge(i, k)
                else:
                    g.remove_edge(k, i)

            d = m.sum(axis=1)  # Aggiornare il vettore dei gradi

    return (g, m)

# Generazione di un grafo casuale con possibilità di controllo sui gradi massimi
def random_graph_with_hub_nodes(n, p, max_degree=None, ratio_hub_nodes=0, print_details=False):
    
    # Controllo di validità dei parametri
    if p < 0 or p > 1:
        print("Error: p must be in [0,1].\n")
        return None

    if n < 0 or (n % 1 != 0):
        print("Error: n must be a positive integer.\n")
        return None
    
    if ratio_hub_nodes < 0 or ratio_hub_nodes > 1:
        print("Error: ratio_popular_nodes must be in [0,1].\n")
        return None
    
    if ratio_hub_nodes != 0 and max_degree is None:
        print("Error: with ratio_popular_nodes choose max_degree.\n")
        return None

    if max_degree is not None:
        if max_degree < 0 or max_degree > n or (max_degree % 1 != 0):
            print("Error: max_degree must be positive integer in [0, n].\n")
            return None

    # Costruzione del vettore che rappresenta tutti i possibili archi
    n_values = int((n**2 - n) / 2)
    edges = [np.random.binomial(1, p) for _ in range(n_values)]

    adjacency_matrix = np.zeros((n, n))  # Inizializzazione della matrice di adiacenza
    Graph = nx.Graph()

    for i in range(n):
        Graph.add_node(i)

    # Costruzione del grafo: solo triangolo superiore
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edges[index] == 1:
                adjacency_matrix[i][j] = 1
                Graph.add_edge(i, j)
            index += 1

    # La matrice di adiacenza viene resa simmetrica per rappresentare un grafo non orientato
    adjacency_matrix += adjacency_matrix.T
    adjacency_matrix = adjacency_matrix.astype(int)

    # Applicazione del vincolo sui gradi se richiesto
    if max_degree is not None:
        before = adjacency_matrix.sum(axis=1)  # Gradi prima della riduzione
        after = before
        pawns = []

        # Se è ammessa una quota di nodi con grado arbitrario
        if 0 < ratio_hub_nodes < 1:
            pawns = choose_nodes_to_adjust(n, before, ratio_hub_nodes, max_degree)
        elif ratio_hub_nodes == 0:
            # Tutti i nodi oltre soglia devono essere ridotti
            for i in range(n):
                if before[i] > max_degree:
                    pawns.append(i)

        # Applicazione della riduzione dei gradi
        if len(pawns) != 0:
            z = degree_reduction(adjacency_matrix, Graph, max_degree, pawns)
            Graph = z[0]
            adjacency_matrix = z[1]
            after = adjacency_matrix.sum(axis=1)

    # Stampa dettagliata del grafo e della sua struttura
    if print_details:
        n_edges = nx.number_of_edges(Graph)
        density = round(nx.density(Graph), 3)

        if max_degree is not None:
            print("\nGradi prima:", before, "\n",
                  "\nGradi dopo: ", after, "\n",
                  "\nArchi spenti =", int(sum(before - after) / 2), "\n")
        else:
            print("\nGradi: ", adjacency_matrix.sum(axis=1), "\n")

        print("Numero di archi:", n_edges, 
              "su", n*(n-1), "possibili.\n\nDensità del grafo:", density,
             "\n\nMatrice di adiacenza (simmetrica):\n", adjacency_matrix,
             "\n\nNodi:", list(Graph.nodes)) 
        print("\nArchi:") 
        pprint.pprint(list(Graph.edges))

    return Graph

def draw_circular_graph(Graph):
    pos = nx.circular_layout(Graph)  # Posizionamento circolare dei nodi

    # Disegno dei nodi (contorno nero, riempimento bianco)
    nx.draw_networkx_nodes(Graph, pos, node_size=500, node_color='white', edgecolors='black')

    # Etichette dei nodi
    nx.draw_networkx_labels(Graph, pos, font_size=10, font_color='black')

    # Disegno degli archi in nero
    nx.draw_networkx_edges(Graph, pos, width=1.2, edge_color='black')

    plt.axis('off')  # Rimozione degli assi cartesiani
    return plt.show()

def draw_compact_bipartite_graph(G):
    # Layout bipartito basato sull'attributo 'bipartite'
    pos = nx.bipartite_layout(G, [node for node in G.nodes if G.nodes[node]['bipartite'] == 0])

    # Disegno dei nodi con contorno nero e riempimento bianco
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')

    # Etichette dei nodi
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    # Disegno degli archi
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='black')

    plt.title("Grafo probabilistico indiretto")
    plt.axis('off')
    plt.show()

# Variante con colorazione
def draw_circular_graph(Graph):
    pos = nx.circular_layout(Graph)  # Posizionamento dei nodi in forma circolare

    # Disegno dei nodi in azzurro chiaro
    nx.draw_networkx_nodes(Graph, pos, node_size=500, node_color='lightblue')

    # Etichette dei nodi
    nx.draw_networkx_labels(Graph, pos, font_size=10)

    # Disegno degli archi
    nx.draw_networkx_edges(Graph, pos, width=1.2, edge_color='lightblue')

    plt.title("Grafo aleatorio indiretto")
    plt.axis('off')
    return plt.show()

# Istogramma della distribuzione dei gradi
def histogram_of_graph_degrees(Graph):
    n_nodes = nx.number_of_nodes(Graph)
    hist = nx.degree_histogram(Graph)  # Frequenza assoluta per ciascun grado

    hist = [x / n_nodes for x in hist]  # Conversione in frequenze relative

    while len(hist) < n_nodes:
        hist.append(0)  # Completamento dell'asse dei gradi fino a n_nodes

    plt.bar(range(n_nodes), hist, align='center')
    plt.ylim(0, max(hist) + 0.01)
    plt.xlabel('Grado')
    plt.ylabel('Frequenza')
    plt.title('Istogramma dei gradi')

    return plt.show()

# Rappresentazione dei sottografi con colori distinti
def draw_subgraphs(Graph):
    forest = list(nx.connected_components(Graph))  # Componenti connesse
    col_dict = {}  # Mappa nodo -> colore
    colors = []  # Palette dei colori
    node_colors = []  # Colore per ciascun nodo

    # Assegnazione di un colore differente a ciascun sottografo
    for i in range(len(forest)):
        colors.append(Blues(0.25 + (i / len(forest)) * 0.75))  # Toni visibili e non troppo chiari

    for i, subgraph in enumerate(forest):
        for node in subgraph:
            col_dict[node] = colors[i]

    pos = nx.spring_layout(Graph)  # Layout a forze per evidenziare cluster

    for node in Graph.nodes():
        node_colors.append(col_dict[node])

    nx.draw(Graph, pos, node_color=node_colors, with_labels=True)
    plt.show()

# Confronto della distribuzione dei gradi per diversi valori di p
def compare_degree_distributions(n, p_values, max_degree = None, ratio_hub_nodes = 0):
    _, axs = plt.subplots(len(p_values), 1)

    plt.suptitle('Distribuzione dei gradi')  # Titolo generale della figura
    max_y = 0  # Massimo valore osservato sull’asse y
    max_x = 0  # Massimo grado osservato sull’asse x

    for i in range(len(p_values)):
        graph = random_graph_with_hub_nodes(n, p_values[i], max_degree, ratio_hub_nodes)

        n_nodes = nx.number_of_nodes(graph)
        hist = nx.degree_histogram(graph)  # Frequenze assolute dei gradi
        hist = [x / n_nodes for x in hist]  # Frequenze relative

        # Aggiornamento dei limiti massimi per assi x e y
        max_x = max(max_x, len(hist))
        max_y = max(max_y, max(hist))

        while len(hist) < (n_nodes):
            hist.append(0)  # Completamento dell’asse x fino al numero totale di nodi

        axs[i].bar(range(n_nodes), hist, align='center')
        axs[i].set_title('n={} p = {} max_degree = {} ratio_hub_nodes = {}'.format(n, p_values[i], max_degree, ratio_hub_nodes), fontsize = 8)

    for ax in axs:
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y + 0.01)

    plt.tight_layout()
    return plt.show()

# Funzioni di verifica per proprietà strutturali del grafo
def connectivity(Graph):
    return nx.is_connected(Graph)

def bipartition(Graph):
    return nx.is_bipartite(Graph)

def aciclicity(Graph):
    return nx.is_forest(Graph)

# Calcolo della frequenza con cui una proprietà è verificata in un campione di grafi
def plot_3d_property_frequency(n_values, p_values, mc = 40, max_degree = None, ratio_hub_nodes = 0, property = '', title = 'Frequenza'):
    
    if property not in [connectivity, bipartition, aciclicity]:
        print('Error: property string is not in\n[connectivity, bipartion, aciclicity].')
        return None

    n_values = sorted(n_values)  # Ordinamento dei parametri
    p_values = sorted(p_values)

    freq = np.zeros((len(n_values), len(p_values)))  # Matrice di frequenze
    one_sample = [0]*mc  # Vettore dei valori osservati della proprietà

    for i in range(len(n_values)):
        print(n_values[i], '---------------------')  # Stato di avanzamento
        for j in range(len(p_values)):
            for k in range(mc):
                g = random_graph_with_hub_nodes(n = n_values[i], p = p_values[j], max_degree = max_degree, ratio_hub_nodes = ratio_hub_nodes)
                one_sample[k] = property(g)  # 1 se il grafo verifica la proprietà

            freq[i][j] = np.mean(one_sample)  # Frequenza relativa della proprietà osservata

    # Costruzione degli assi per il grafico 3D
    x, y = np.meshgrid(p_values, n_values)
    z = freq

    fig = plt.figure()
    gr = fig.add_subplot(111, projection='3d')
    gr.plot_surface(x, y, z, cmap='viridis')

    gr.set_xlabel('p')
    gr.set_ylabel('n')
    gr.set_zlabel('Frequenza')
    gr.set_title(title)

    return plt.show()

# Scrittura su file dei grafi generati con parametri specifici
def write_graphs_on_file(n_list, p_list, max_degree = None, ratio_hub_nodes = 0, mc = 40, file_name='grafi.txt'):
    start = time.time()

    n_list = sorted(n_list)
    p_list = sorted(p_list)

    with open(file_name, 'wb') as file:
        for n in n_list:
            print(n, '---------------------------')

            if max_degree == None:
                md = n-1
            else:
                md = max_degree

            for p in p_list:
                n = int(n)
                p = round(p, 2)
                file.write(f'{n} {p} {md} {ratio_hub_nodes} {mc}\n'.encode())
                for k in range(mc):  
                    file.write(f'{k}\n'.encode())
                    g = random_graph_with_hub_nodes(n, p, max_degree, ratio_hub_nodes)
                    nx.write_edgelist(g, file, data=False) 

    stop = time.time()
    print('Run time: ', (stop-start), 'seconds')
    return file_name

# Lettura dei grafi salvati su file e ricostruzione in memoria
def read_graphs_from_file(file_name):
    try:
        with open(file_name, 'rb') as file:
            g_dict = {}  # Dizionario dei grafi
            g_list = []  # Lista dei grafi per ogni combinazione n, p
            g = None
            n = None
            p = None

            for l in file:
                l = l.decode().strip()
                d = l.split()

                if len(d) == 5:  # Inizio di un nuovo blocco di grafi
                    if g is not None:
                        g_list.append(g)
                        g_dict[(n, p, md, rhn, mc)] = g_list

                    n, p, md, rhn, mc = map(float, d)
                    g_list = []
                    g = None

                else:
                    if len(d) == 1:  # Inizio di un nuovo grafo
                        g_list.append(g)
                        g = nx.Graph()
                        for i in range(int(n)):
                            g.add_node(i)

                    if len(d) == 2:  # Aggiunta di un arco
                        u, v = map(int, d)
                        g.add_edge(u, v)

            if n is not None and p is not None:
                g_dict[(n, p, md, rhn, mc)] = g_list

        return g_dict

    except FileNotFoundError:
        return None

#--------------------------------------------------------------------------------------
# Interfaccia utente – gestione delle sezioni di input e output
#--------------------------------------------------------------------------------------

### Sezione 1: Generazione e analisi di un singolo grafo

def section_single_graph():
    # Estrazione dei parametri dalla GUI
    n = int(n_par1.get())
    p = float(p_par1.get())
    
    max_degree_str = md_par1.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par1.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    details = bool(det_par_var1.get())
    g = random_graph_with_hub_nodes(n, p, max_degree, ratio_hub_nodes, details)

    return g

def section_single_graph_property():
    # Generazione del grafo con parametri utente
    g = section_single_graph()
    
    # Visualizzazione delle proprietà del grafo nella GUI
    result_label1.config(text=f"Connessione: {connectivity(g)}\nAciclicità: {aciclicity(g)}\nBipartizione: {bipartition(g)}")
    
    # Selezione del tipo di visualizzazione da menu
    options_onegraph = menu_onegraph1.get()
    
    if options_onegraph == "Grafico circolare":
        draw_circular_graph(g)
    
    if options_onegraph == "Rappresentazione dei sottografi":
        draw_subgraphs(g)
    
    if options_onegraph == "Istogramma dei gradi":
        histogram_of_graph_degrees(g)

### Sezione 2: Salvataggio di più grafi su file

def section_writing_on_file():
    # Acquisizione dei parametri da input testuali multipli (n e p)
    n_values = n_val2.get()
    n_list = [int(x.strip()) for x in n_values.split(',')]
    
    p_values = p_val2.get()
    p_list = [float(x.strip()) for x in p_values.split(',')]
    
    max_degree_str = md_par2.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par2.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0
    
    mc = int(mc_par2.get())
    file_write = str(fw_str2.get())
    
    # Scrittura su file dei grafi generati con le combinazioni specificate
    write_graphs_on_file(n_list, p_list, max_degree, ratio_hub_nodes, mc, file_write)

### Sezione 3: Lettura di grafi da file e visualizzazione

def section_read_file1():
    file_read = str(fr_str3.get())
    dict = read_graphs_from_file(file_read)

    # Controllo dell'esistenza del file
    if dict == None:
        result_label3.config(text=f"Il file {file_read} non esiste")
        return None
    
    par = dict.keys()
    n_val = []
    p_val = []

    # Estrazione e salvataggio dei parametri unici n e p presenti nel file
    for x in par:
        n, p, md, rhn, mc = x
        if n not in n_val:
            n_val.append(int(n))
        if p not in p_val:
            p_val.append(p)
    
    # Visualizzazione dei parametri dei grafi letti
    if md == (max(n_val)-1):
        result_label3.config(text="Il file contiene grafi generati dai seguenti parametri:\nn: {}\np: {}\nmd: {}\nrhn: {}\nmc: {}".format(n_val, p_val, None, rhn, int(mc)))
    else:
        result_label3.config(text="Il file contiene grafi generati dai seguenti parametri:\nn: {}\np: {}\nmd: {}\nrhn: {}\nmc: {}".format(n_val, p_val, int(md), rhn, int(mc)))
    
    return dict

def section_read_file2():
    dict = section_read_file1()
    g = None

    # Estrazione dei parametri n e p da tutte le combinazioni presenti
    all_keys = dict.keys()
    n_val = []
    p_val = []

    for x in all_keys:
        n, p, md, rhn, mc = x
        if n not in n_val:
            n_val.append(int(n))
        if p not in p_val:
            p_val.append(p)

    # Parsing dei parametri scelti da utente
    choose_par = str(par_str3.get())  
    par = [float(item.strip()) for item in choose_par.split(',')]
    
    # Ricerca della tupla di parametri corrispondente
    for one_key in all_keys:
        if one_key[:2] == par:
            par = one_key

    choose_mc = int(mc_num3.get())

    # Estrazione del grafo specifico
    g = dict[one_key][choose_mc]

    # Visualizzazione delle proprietà del grafo
    result_label33.config(text=f"Connessione: {connectivity(g)}\nAciclicità: {aciclicity(g)}\nBipartizione: {bipartition(g)}")
    
    # Visualizzazione grafica del grafo selezionato
    options_drawgraph = menu_drawgraph3.get()
    
    if options_drawgraph == "Grafico circolare":
        draw_circular_graph(g)
    
    if options_drawgraph == "Rappresentazione dei sottografi":
        draw_subgraphs(g)
    
    if options_drawgraph == "Istogramma dei gradi":
        histogram_of_graph_degrees(g)

### Sezione 4: Visualizzazione 3D delle frequenze delle proprietà

def section_histogram_of_freq():
    # Acquisizione dei valori di n e p da input utente
    n_values = n_val4.get()
    n_list = [int(x.strip()) for x in n_values.split(',')]
    
    p_values = p_val4.get()
    p_list = [float(x.strip()) for x in p_values.split(',')]

    # Parametri opzionali
    max_degree_str = md_par4.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    ratio_hub_nodes_str = rhn_par4.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    mc = int(mc_par4.get())
    
    # Selezione della proprietà da analizzare
    property_3dimplot = menu_property4.get()
    res = None

    # Plot della frequenza in base alla proprietà scelta
    if property_3dimplot == "Connettività":
        title = 'Frequenza di grafi connessi'
        plot_3d_property_frequency(n_list, p_list, mc, max_degree, ratio_hub_nodes, property=connectivity, title=title)

    if property_3dimplot == "Aciclicità":
        title = 'Frequenza di grafi aciclici'
        plot_3d_property_frequency(n_list, p_list, mc, max_degree, ratio_hub_nodes, property=aciclicity, title=title)

    if property_3dimplot == "Bipartizione":
        title = 'Frequenza di grafi bipartiti'
        plot_3d_property_frequency(n_list, p_list, mc, max_degree, ratio_hub_nodes, property=bipartition, title=title)

### Sezione 5: Confronto delle distribuzioni dei gradi

def section_compare_degree_distribution():
    # Acquisizione del parametro n da input utente
    n = int(n_par5.get())

    # Parsing della lista di probabilità p
    p_values = p_val5.get()  
    p_list = [float(x.strip()) for x in p_values.split(',')] 
    
    # Acquisizione del grado massimo (opzionale)
    max_degree_str = md_par5.get().strip()
    max_degree = int(max_degree_str) if max_degree_str else None
    
    # Acquisizione del rapporto dei nodi hub (opzionale)
    ratio_hub_nodes_str = rhn_par5.get().strip()
    ratio_hub_nodes = float(ratio_hub_nodes_str) if ratio_hub_nodes_str else 0.0

    # Richiamo alla funzione di confronto delle distribuzioni di grado
    compare_degree_distributions(n, p_list, max_degree, ratio_hub_nodes)

#### Sezione 3: Acquisizione da file ------------------------------------------

# pt1: Creazione del frame per la Sezione 3 (Acquisizione da file)
section3 = ttk.Frame(notebook)
notebook.add(section3, text='Acquisizione da file')

# Etichetta e campo di input per il nome del file da cui acquisire i dati
fr_str_label3 = ttk.Label(section3, text="Nome del file:")
fr_str_label3.pack()
fr_str3 = ttk.Entry(section3)
fr_str3.pack()

# Pulsante per eseguire la lettura dei dati dal file
execute_write_button3 = ttk.Button(section3, text="Esegui", command=section_read_file1)
execute_write_button3.pack()

# Etichetta per visualizzare i risultati dell'acquisizione dal file
result_label3 = ttk.Label(section3, text="")
result_label3.pack()

# pt2: Creazione della seconda parte della Sezione 3 (Selezione dei parametri)
# Etichetta e campo di input per selezionare i parametri da acquisire (n, p)
par_str_label3 = ttk.Label(section3, text="Parametri da selezionare (n,p):")
par_str_label3.pack()
par_str3 = ttk.Entry(section3)
par_str3.pack()

# Etichetta e campo di input per selezionare l'elemento specifico (mc_num)
mc_num_label3 = ttk.Label(section3, text="Elemento:")
mc_num_label3.pack()
mc_num3 = ttk.Entry(section3)
mc_num3.pack()

# Menu a tendina per selezionare il tipo di grafico da visualizzare
menu_drawgraph_label3 = ttk.Label(section3, text="Tipo di grafico:")
menu_drawgraph_label3.pack()
menu_drawgraph3 = ttk.Combobox(section3, values=("Grafico circolare", "Rappresentazione dei sottografi", "Istogramma dei gradi"))
menu_drawgraph3.pack()

# Pulsante per eseguire l'acquisizione del grafo in base ai parametri selezionati
execute_read_button3 = ttk.Button(section3, text="Esegui", command=section_read_file2)
execute_read_button3.pack()

# Etichetta per visualizzare i risultati dell'acquisizione dei grafi
result_label33 = ttk.Label(section3, text="")
result_label33.pack()

### Sezione 4: Grafico tridimensionale ------------------------------------------

# Creazione del frame per la Sezione 4 (Grafico tridimensionale)
section4 = ttk.Frame(notebook)
notebook.add(section4, text='Grafico tridimensionale')

# Etichetta e campo di input per inserire la lista di valori n da confrontare
n_val_label4 = ttk.Label(section4, text="Lista di valori n da confrontare (separati da virgola):")
n_val_label4.pack()
n_val4 = ttk.Entry(section4)
n_val4.pack()

# Etichetta e campo di input per inserire la lista di valori p da confrontare
p_val_label4 = ttk.Label(section4, text="Lista di valori p da confrontare (separati da virgola):")
p_val_label4.pack()
p_val4 = ttk.Entry(section4)
p_val4.pack()

# Etichetta e campo di input per il grado massimo
md_par_label4 = ttk.Label(section4, text="Grado massimo:")
md_par_label4.pack()
md_par4 = ttk.Entry(section4)
md_par4.pack()

# Etichetta e campo di input per la proporzione di nodi centrali
rhn_par4_label = ttk.Label(section4, text="Proporzione di nodi centrali:")
rhn_par4_label.pack()
rhn_par4 = ttk.Entry(section4)
rhn_par4.pack()

# Etichetta e campo di input per il numero di campioni
mc_par4_label = ttk.Label(section4, text="Numero di campioni:")
mc_par4_label.pack()
mc_par4 = ttk.Entry(section4)
mc_par4.pack()

# Menu a tendina per selezionare la proprietà da verificare (connettività, aciclicità, bipartizione)
menu_property_label4 = ttk.Label(section4, text="Verifica proprietà:")
menu_property_label4.pack()
menu_property4 = ttk.Combobox(section4, values=("Connettività", "Aciclicità", "Bipartizione"))
menu_property4.pack()

# Pulsante per eseguire il calcolo del grafico tridimensionale in base alle proprietà selezionate
execute_write_button4 = ttk.Button(section4, text="Esegui", command=section_histogram_of_freq)
execute_write_button4.pack()

##### Sezione 5: Confronto tra distribuzioni dei gradi ------------------------------------------

# Creazione del frame per la Sezione 5 (Confronto tra distribuzioni dei gradi)
section5 = ttk.Frame(notebook)
notebook.add(section5, text='Distribuzione dei gradi')

# Etichetta e campo di input per il numero di nodi
n_par_label5 = ttk.Label(section5, text="Numero di nodi:")
n_par_label5.pack()
n_par5 = ttk.Entry(section5)
n_par5.pack()

# Etichetta e campo di input per inserire la lista di valori p da confrontare
p_val_label5 = ttk.Label(section5, text="Lista di valori p da confrontare: (separati da virgola)")
p_val_label5.pack()
p_val5 = ttk.Entry(section5)
p_val5.pack()

# Etichetta e campo di input per il grado massimo
md_par_label5 = ttk.Label(section5, text="Grado massimo:")
md_par_label5.pack()
md_par5 = ttk.Entry(section5)
md_par5.pack()

# Etichetta e campo di input per la proporzione di nodi centrali
rhn_par_label5 = ttk.Label(section5, text="Proporzione di nodi centrali:")
rhn_par_label5.pack()
rhn_par5 = ttk.Entry(section5)
rhn_par5.pack()

# Pulsante per eseguire il confronto tra distribuzioni dei gradi
execute_button5 = ttk.Button(section5, text="Esegui", command=section_compare_degree_distribution)
execute_button5.pack()

# Avvia l'applicazione principale
app.mainloop()
