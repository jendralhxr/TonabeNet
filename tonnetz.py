import networkx as nx
import matplotlib.pyplot as plt

MIN_NOTE = 21  # C2
MAX_NOTE = 108  # C8

def midi_to_pitch(m):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{pitch_classes[m % 12]}{m // 12}"

def midi_to_coord(midi):
    # Express pitch as combination of 3 and 4 semitone steps (m3, M3, P5 lattice)
    q = midi // 4    # rough: count of M3 steps
    r = midi // 3    # rough: count of m3 steps
    return (q, r)

INTERVALS = {
    'M3': 4,   # horizontal
    'P5': 7,   # down-right
    'm3': 3,   # up-right
}

DIRECTIONS = {
    'M3': (1, 0.5),   # right
    'P5': (0, 1),     # down-right
    'm3': (1, -0.5),  # up-right
}

G = nx.Graph()
positions = {}
coord_to_names = {}   # collect multiple note names per coordinate
visited = set()

queue = [(21, (0, 0))]  # Start at C4

# for midi in range(MIN_NOTE, MAX_NOTE+1):
#     coord = midi_to_coord(midi)
#     positions[coord] = (coord[0], -coord[1])
#     name = midi_to_pitch(midi)
#     coord_to_names.setdefault(coord, []).append(name)

while queue:
    midi, (q, r) = queue.pop(0)
    if not (MIN_NOTE <= midi <= MAX_NOTE) or midi in visited:
        continue

    visited.add(midi)
    name = midi_to_pitch(midi)

    # merge nodes by coordinate
    if (q, r) not in coord_to_names:
        coord_to_names[(q, r)] = []
    coord_to_names[(q, r)].append(name)

    positions[(q, r)] = (q, r) # higher pitch up
    # positions[(q, r)] = (q, -r) higher pitch down

    for label, interval in INTERVALS.items():
        next_midi = midi + interval
        dq, dr = DIRECTIONS[label]
        neighbor_coord = (q + dq, r + dr)
        if MIN_NOTE <= next_midi <= MAX_NOTE:
            queue.append((next_midi, neighbor_coord))

# Build graph with merged nodes
coord_to_label = {coord: "\n".join(names) for coord, names in coord_to_names.items()}

for coord, label in coord_to_label.items():
    G.add_node(label)
    positions[label] = positions[coord]

# Add edges (check merged labels!)
for coord, names in coord_to_names.items():
    label = coord_to_label[coord]
    midi = names[0]  # representative
    for intvl, semitones in INTERVALS.items():
        dq, dr = DIRECTIONS[intvl]
        neighbor_coord = (coord[0] + dq, coord[1] + dr)
        if neighbor_coord in coord_to_label:
            neighbor_label = coord_to_label[neighbor_coord]
            if not G.has_edge(label, neighbor_label):
                G.add_edge(label, neighbor_label, interval=intvl)

# Split edges by type
p5_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interval'] == 'P5']
m3_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interval'] == 'm3']
M3_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interval'] == 'M3']

plt.figure(figsize=(10, 16))

# Draw nodes
nx.draw_networkx_nodes(G, pos=positions, node_size=1400, node_color='yellow')
nx.draw_networkx_labels(G, pos=positions, font_size=12)

# Draw edges with styles
nx.draw_networkx_edges(G, pos=positions, edgelist=p5_edges,
                       edge_color='blue', width=4, style='solid')
nx.draw_networkx_edges(G, pos=positions, edgelist=M3_edges,
                       edge_color='green', width=3, style='dashed')
nx.draw_networkx_edges(G, pos=positions, edgelist=m3_edges,
                       edge_color='red', width=2, style='dashed')

nx.draw_networkx_edges(
    G, pos=positions, edgelist=p5_edges,
    edge_color='blue', width=4, style='solid',
    label="Perfect 5th"
)
nx.draw_networkx_edges(
    G, pos=positions, edgelist=M3_edges,
    edge_color='green', width=3, style='dashed',
    label="Major 3rd"
)
nx.draw_networkx_edges(
    G, pos=positions, edgelist=m3_edges,
    edge_color='red', width=2, style='dashed',
    label="Minor 3rd"
)

# let networkx build the legend box
plt.legend(loc="lower left")

plt.title("Tonnetz (Merged Enharmonic/Overlapping Nodes)", fontsize=14)
plt.axis('off')
plt.show()
