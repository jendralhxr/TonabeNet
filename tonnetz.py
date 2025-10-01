import sys
import networkx as nx
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtCore import Qt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PySide6.QtCore import QRectF

# ----------------------------
# Tonnetz Graph Construction
# ----------------------------

MIN_NOTE = 21  # C2
MAX_NOTE = 108  # C8

def midi_to_pitch(m):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{pitch_classes[m % 12]}{m // 12}"

INTERVALS = {
    'M3': 4,   # major 3rd
    'P5': 7,   # perfect 5th
    'm3': 3,   # minor 3rd
}

DIRECTIONS = {
    'M3': (1, 0.5),   # right
    'P5': (0, 1),     # down-right
    'm3': (1, -0.5),  # up-right
}

G = nx.Graph()
positions = {}
coord_to_names = {}   # multiple note names per coordinate
visited = set()
queue = [(MIN_NOTE, (0, 0))]  # Start at C2

while queue:
    midi, (q, r) = queue.pop(0)
    if not (MIN_NOTE <= midi <= MAX_NOTE) or midi in visited:
        continue

    visited.add(midi)
    name = midi_to_pitch(midi)

    if (q, r) not in coord_to_names:
        coord_to_names[(q, r)] = []
    coord_to_names[(q, r)].append(name)

    positions[(q, r)] = (q, r)

    for label, interval in INTERVALS.items():
        next_midi = midi + interval
        dq, dr = DIRECTIONS[label]
        neighbor_coord = (q + dq, r + dr)
        if MIN_NOTE <= next_midi <= MAX_NOTE:
            queue.append((next_midi, neighbor_coord))

# merge node names by coordinate
coord_to_label = {coord: "\n".join(names) for coord, names in coord_to_names.items()}

for coord, label in coord_to_label.items():
    G.add_node(label)
    positions[label] = positions[coord]

for coord, names in coord_to_names.items():
    label = coord_to_label[coord]
    for intvl, semitones in INTERVALS.items():
        dq, dr = DIRECTIONS[intvl]
        neighbor_coord = (coord[0] + dq, coord[1] + dr)
        if neighbor_coord in coord_to_label:
            neighbor_label = coord_to_label[neighbor_coord]
            if not G.has_edge(label, neighbor_label):
                G.add_edge(label, neighbor_label, interval=intvl)

# Initialize node values (all start at 0)
node_values = {label: 0 for label in G.nodes()}

# ----------------------------
# Tonnetz Widget
# ----------------------------

class TonnetzWidget(QWidget):
    def __init__(self, G, positions, node_values, parent=None):
        super().__init__(parent)
        self.G = G
        self.positions = positions
        self.node_values = node_values

        self.cmap = cm.get_cmap("rainbow")
        max_val = max(node_values.values()) if node_values else 1
        self.norm = mcolors.Normalize(vmin=0, vmax=max_val)

        self.resize(360, 900)
        self.setWindowTitle("Tonnetz Widget (63 nodes)")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        scale = 60
        offset_x, offset_y = 60,750

        # Draw edges
        for u, v, d in self.G.edges(data=True):
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            x1, y1 = x1 * scale + offset_x, -y1 * scale + offset_y
            x2, y2 = x2 * scale + offset_x, -y2 * scale + offset_y

            if d["interval"] == "P5":
                pen = QPen(QColor("blue"), 3, Qt.SolidLine)
            elif d["interval"] == "M3":
                pen = QPen(QColor("green"), 3, Qt.DashLine)
            else:  # m3
                pen = QPen(QColor("red"), 3, Qt.DotLine)

            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)

        # Draw nodes
        radius = 20
        font = QFont("Monospace", 10)
        painter.setFont(font)

        for n in self.G.nodes():
            x, y = self.positions[n]
            x, y = x * scale + offset_x, -y * scale + offset_y

            val = self.node_values.get(n, 0)
            rgba = self.cmap(self.norm(val))
            color = QColor.fromRgbF(*rgba)

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(x - radius, y - radius, 2*radius, 2*radius)

            # Label
            painter.setPen(QPen(QColor("white")))
            # painter.drawText(x - (radius/2), y, n)
            # painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, n)
            rect = QRectF(x - radius, y - radius -2, 2*radius, 2*radius)
            painter.setPen(QPen(QColor("white")))
            painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, n)



# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TonnetzWidget(G, positions, node_values)
    w.show()
    sys.exit(app.exec())
