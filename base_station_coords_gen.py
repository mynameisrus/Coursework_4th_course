import json
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import random


class HexagonalNetwork:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.ISD = self.config["network_parameters"]["ISD_m"]
        self.tiers = self.config["network_parameters"]["tiers"]
        self.total_users = self.config["network_parameters"].get("total_users", 0)
        self.cell_radius = self.ISD / math.sqrt(3)

    def load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def hex_to_cartesian(self, x: int, y: int, z: int) -> Tuple[float, float]:
        X = (x + y / 2) * self.ISD
        Y = (y * math.sqrt(3) / 2) * self.ISD
        return (X, Y)

    def generate_bs_coordinates(self) -> List[Tuple[float, float, int]]:
        coords = [(0.0, 0.0, 0)]

        directions = [
            (1, -1, 0),
            (1, 0, -1),
            (0, 1, -1),
            (-1, 1, 0),
            (-1, 0, 1),
            (0, -1, 1),
        ]

        for k in range(1, self.tiers + 1):
            for side in range(6):
                x = k * directions[side][0]
                y = k * directions[side][1]
                z = k * directions[side][2]

                dir_x = directions[(side + 2) % 6][0]
                dir_y = directions[(side + 2) % 6][1]
                dir_z = directions[(side + 2) % 6][2]

                for step in range(k):
                    X, Y = self.hex_to_cartesian(x, y, z)
                    coords.append((X, Y, k))

                    x += dir_x
                    y += dir_y
                    z += dir_z

        return coords

    def generate_user_coordinates(self) -> List[Tuple[float, float, int]]:
        if self.total_users <= 0:
            return []

        bs_coords = self.generate_bs_coordinates()
        user_coords = []

        for user_id in range(self.total_users):
            bs_index = random.randint(0, len(bs_coords) - 1)
            bs_x, bs_y, _ = bs_coords[bs_index]
            sector = random.randint(0, 5)

            r1 = random.random()
            r2 = random.random()

            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2

            angle_offset = math.pi / 6

            angle_base = sector * math.pi / 3 + angle_offset
            angle_next = (sector + 1) * math.pi / 3 + angle_offset

            x1, y1 = 0, 0
            x2, y2 = self.cell_radius * math.cos(angle_base), self.cell_radius * math.sin(angle_base)
            x3, y3 = self.cell_radius * math.cos(angle_next), self.cell_radius * math.sin(angle_next)

            user_rel_x = r1 * x2 + r2 * x3 + (1 - r1 - r2) * x1
            user_rel_y = r1 * y2 + r2 * y3 + (1 - r1 - r2) * y1

            user_x = bs_x + user_rel_x
            user_y = bs_y + user_rel_y

            user_coords.append((user_x, user_y, bs_index))

        return user_coords

    def visualize(self):
        coords = self.generate_bs_coordinates()
        user_coords = self.generate_user_coordinates()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        if user_coords:
            user_x = [x for x, y, _ in user_coords]
            user_y = [y for x, y, _ in user_coords]
            ax.scatter(user_x, user_y, c='red', s=30, alpha=0.7, marker='o', zorder=5)

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
        ax.set_title(f'Гексагональная сеть (*R3): ISD={self.ISD} м, Tiers={self.tiers}, Всего БС={len(coords)}',fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    network = HexagonalNetwork("config.json")
    network.visualize()


if __name__ == "__main__":
    main()