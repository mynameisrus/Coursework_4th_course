import json
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


class HexagonalNetwork:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.ISD = self.config["network_parameters"]["ISD_m"]
        self.tiers = self.config["network_parameters"]["tiers"]
        self.cell_radius = self.ISD / math.sqrt(3)

    def load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_bs_coordinates(self) -> List[Tuple[float, float, int]]:
        coords = [(0.0, 0.0, 0)]
        for k in range(1, self.tiers + 1):
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    if abs(i) + abs(j) + abs(i + j) <= 2 * k and not (i == 0 and j == 0):
                        x = (i + j / 2) * self.ISD
                        y = (j * math.sqrt(3) / 2) * self.ISD
                        tier = max(abs(i), abs(j), abs(i + j))
                        coords.append((x, y, tier))
        return coords

    def visualize(self):
        coords = self.generate_bs_coordinates()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        for x, y, tier in coords:
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=self.cell_radius,
                orientation=0,
                edgecolor='black',
                alpha=0.5,
                linewidth=1.2
            )
            ax.add_patch(hexagon)

        for x, y, tier in coords:
            ax.plot(x, y, 'ko', markersize=8, zorder=10)

        max_range = (self.tiers + 1) * self.ISD * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X (м)', fontsize=12)
        ax.set_ylabel('Y (м)', fontsize=12)
        ax.set_title(f'Гексагональная сеть: ISD={self.ISD} м, Tiers={self.tiers}, Всего БС={len(coords)}',fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    network = HexagonalNetwork("config.json")
    network.visualize()

if __name__ == "__main__":
    main()