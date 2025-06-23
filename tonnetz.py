# tonnetz lattice grid
# Horizontal axis: perfect fifths (P5)
# Diagonal axes: major thirds (M3) and minor thirds (m3)

import networkx as nx
import matplotlib.pyplot as plt

# MIDI notes for 7 octaves (e.g. MIDI 36 = C2, up to MIDI 108 = C8)
MIN_NOTE = 36
MAX_NOTE = 108

# Map MIDI numbers to pitch names with octaves
def midi_to_pitch(m):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{pitch_classes[m % 12]}{m // 12}"

# Intervals in semitones
INTERVALS = {
    'P5': 7,   # perfect fifth
    'M3': 4,   # major third
    'm3': 3,   # minor third
}

# Directions for hex layout
# (q, r) coordinates like axial hex grid
DIRECTIONS = {
    'P5': (1, 0),
    'M3': (0, 1),
    'm3': (1, -1),
}

G = nx.Graph()
positions = {}  # hex layout positions
midi_to_coord = {}  # to prevent duplicates

# Start at center with C4 (MIDI 60)
center = (0, 0)
start_note = 60
queue = [(start_note, center)]
visited = set()

while queue:
    midi, (q, r) = queue.pop(0)
    if midi < MIN_NOTE or midi > MAX_NOTE:
        continue
    if midi in visited:
        continue

    visited.add(midi)
    name = midi_to_pitch(midi)
    G.add_node(name)
    positions[name] = (q, -r)  # Flip y to make plot upward

    midi_to_coord[midi] = (q, r)

    for label, interval in INTERVALS.items():
        neighbor_midi = midi + interval
        dq, dr = DIRECTIONS[label]
        neighbor_coord = (q + dq, r + dr)
        if neighbor_midi not in visited:
            queue.append((neighbor_midi, neighbor_coord))
        if MIN_NOTE <= neighbor_midi <= MAX_NOTE:
            neighbor_name = midi_to_pitch(neighbor_midi)
            G.add_edge(name, neighbor_name, interval=label)

# Draw the graph
color_map = {'P5': 'blue', 'M3': 'green', 'm3': 'red'}
edge_colors = [color_map[G[u][v]['interval']] for u, v in G.edges]

plt.figure(figsize=(14, 12))
plt.figure()
nx.draw(G, pos=positions, with_labels=True, node_size=90,
        node_color='lightyellow', font_size=8,
        edge_color=edge_colors, width=1.8)

plt.title("Tonnetz (Hex Lattice, 7 Octaves)", fontsize=16)
plt.axis('off')
plt.show()
