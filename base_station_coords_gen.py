import json
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import random


class HexagonalNetwork:
    HEX_DIRECTIONS = [
        (1, 0, -1), (0, 1, -1), (-1, 1, 0),
        (-1, 0, 1), (0, -1, 1), (1, -1, 0),
    ]

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()

        self.ISD = self.config["network_parameters"]["ISD_m"]
        self.tiers = self.config["network_parameters"].get("tiers", 2)
        self.mode = self.config["network_parameters"].get("mode", "tiers")
        self.site_radius = self.ISD / math.sqrt(3)
        self.total_users = self.config["network_parameters"].get("total_users", 0)
        if self.mode == 1:
            self.cell_radius = self.site_radius / math.sqrt(3)
        else:
            self.cell_radius = self.site_radius

    def load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def hex_to_cartesian(self, x: int, y: int, z: int) -> Tuple[float, float]:
        X = (x + y / 2) * self.ISD
        Y = (y * math.sqrt(3) / 2) * self.ISD

        if self.mode == 1:
            angle = math.pi / 6
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            X_rot = X * cos_a - Y * sin_a
            Y_rot = X * sin_a + Y * cos_a
            return (X_rot, Y_rot)

        return (X, Y)

    def _generate_hex_grid(self) -> List[Tuple[float, float, int]]:
        coords = [(0.0, 0.0, 0)]

        for tier in range(1, self.tiers + 1):
            for side in range(6):
                for step in range(tier):
                    x = (tier - step) * self.HEX_DIRECTIONS[side][0] + step * self.HEX_DIRECTIONS[(side + 1) % 6][0]
                    y = (tier - step) * self.HEX_DIRECTIONS[side][1] + step * self.HEX_DIRECTIONS[(side + 1) % 6][1]
                    z = (tier - step) * self.HEX_DIRECTIONS[side][2] + step * self.HEX_DIRECTIONS[(side + 1) % 6][2]

                    X, Y = self.hex_to_cartesian(x, y, z)
                    coords.append((X, Y, tier))

        return coords

    def generate_bs_coordinates(self) -> List[Tuple[float, float, int]]:
        if self.mode == 1:
            return self.generate_tri_hex_centers()
        else:
            return self._generate_hex_grid()

    def generate_tri_hex_centers(self) -> List[Tuple[float, float, int]]:
        bs_coords = self._generate_hex_grid()
        hex_centers = []
        seen_centers = set()

        hex_angles = [math.pi / 6, 5 * math.pi / 6, 3 * math.pi / 2]

        for bs_x, bs_y, tier in bs_coords:
            for angle in hex_angles:
                center_x = bs_x + self.cell_radius * math.cos(angle)
                center_y = bs_y + self.cell_radius * math.sin(angle)

                hex_centers.append((center_x, center_y, tier))

        return hex_centers

    def get_bs_positions(self) -> List[Tuple[float, float, int]]:
        return self._generate_hex_grid()

    def generate_user_coordinates(self) -> List[Tuple[float, float, int]]:
        if self.total_users <= 0:
            return []

        hex_coords = self.generate_bs_coordinates()

        if len(hex_coords) == 0:
            return []

        user_coords = []

        for user_id in range(self.total_users):
            hex_index = random.randint(0, len(hex_coords) - 1)
            hex_x, hex_y, _ = hex_coords[hex_index]
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

            user_x = hex_x + user_rel_x
            user_y = hex_y + user_rel_y

            user_coords.append((user_x, user_y, hex_index))

        return user_coords

    def visualize(self):
        hex_coords = self.generate_bs_coordinates()
        bs_coords = self.get_bs_positions()
        user_coords = self.generate_user_coordinates()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')

        if user_coords:
            user_x = [x for x, y, _ in user_coords]
            user_y = [y for x, y, _ in user_coords]
            ax.scatter(user_x, user_y, c='red', s=30, alpha=0.7, marker='o', zorder=5)

        for x, y, tier in hex_coords:
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

        for x, y, tier in bs_coords:
            ax.plot(x, y, 'ko', markersize=8, zorder=10)

        max_range = (self.tiers + 1) * self.ISD * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X (м)', fontsize=12)
        ax.set_ylabel('Y (м)', fontsize=12)

        num_bs = len(bs_coords)
        num_hex = len(hex_coords)
        ax.set_title(f'Гексагональная сеть: ISD={self.ISD} м',fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    network = HexagonalNetwork("config.json")
    network.visualize()


if __name__ == "__main__":
    main()