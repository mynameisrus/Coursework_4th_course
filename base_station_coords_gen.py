import json
import math
import csv
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
        self.tiers = self.config["network_parameters"].get("tiers", 1)
        self.mode = self.config["network_parameters"].get("mode", 1)
        self.site_radius = self.ISD / math.sqrt(3)
        self.total_users = self.config["network_parameters"].get("total_users", 0)
        self.bs_height = self.config["network_parameters"].get("bs_height", 20)
        self.wrap_tiers = self.config["network_parameters"].get("wrap_tiers", 1)

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
            cos_a = math.cos(math.pi / 6)
            sin_a = math.sin(math.pi / 6)
            X_rot = X * cos_a - Y * sin_a
            Y_rot = X * sin_a + Y * cos_a
            return (X_rot, Y_rot)

        return (X, Y)

    def generate_hex_grid(self) -> List[Tuple[float, float, float, int]]:
        coords = [(0.0, 0.0, self.bs_height, 0)]

        for tier in range(1, self.tiers + 1):
            for side in range(6):
                for step in range(tier):
                    x = (tier - step) * self.HEX_DIRECTIONS[side][0] + step * self.HEX_DIRECTIONS[(side + 1) % 6][0]
                    y = (tier - step) * self.HEX_DIRECTIONS[side][1] + step * self.HEX_DIRECTIONS[(side + 1) % 6][1]
                    z = (tier - step) * self.HEX_DIRECTIONS[side][2] + step * self.HEX_DIRECTIONS[(side + 1) % 6][2]

                    X, Y = self.hex_to_cartesian(x, y, z)
                    Z = self.bs_height
                    coords.append((X, Y, Z, tier))

        return coords

    def generate_bs_coordinates(self) -> List[Tuple[float, float, float, int]]:
        if self.mode == 1:
            return self.generate_tri_hex_centers()
        else:
            return self.generate_hex_grid()

    def generate_tri_hex_centers(self) -> List[Tuple[float, float, float, int]]:
        bs_coords = self.generate_hex_grid()
        hex_centers = []
        hex_angles = [math.pi / 6, 5 * math.pi / 6, 3 * math.pi / 2]

        for bs_x, bs_y, bs_z, tier in bs_coords:
            for angle in hex_angles:
                center_x = bs_x + self.cell_radius * math.cos(angle)
                center_y = bs_y + self.cell_radius * math.sin(angle)
                hex_centers.append((center_x, center_y, bs_z, tier))

        return hex_centers

    def get_bs_positions(self) -> List[Tuple[float, float, float, int]]:
        return self.generate_bs_coordinates()

    def generate_wrapped_bs_positions(self) -> List[Tuple[float, float, float, int, int]]:
        central_bs = self.get_bs_positions()
        all_bs = []

        for x, y, z, tier in central_bs:
            all_bs.append((x, y, z, tier, 0))

        wrap_directions = self.HEX_DIRECTIONS
        shift_distance = (2 * self.tiers + 1) * self.ISD

        copy_offsets = [
            (-self.ISD / 2, self.ISD * math.sqrt(3) / 2),
            (-self.ISD, 0),
            (-self.ISD / 2, -self.ISD * math.sqrt(3) / 2),
            (self.ISD / 2, -self.ISD * math.sqrt(3) / 2),
            (self.ISD, 0),
            (self.ISD / 2, self.ISD * math.sqrt(3) / 2)
        ]

        for wrap_id, (dx, dy, dz) in enumerate(wrap_directions, start=1):
            shift_x = (dx + dy / 2) * shift_distance
            shift_y = (dy * math.sqrt(3) / 2) * shift_distance

            offset_x, offset_y = copy_offsets[wrap_id - 1]
            shift_x += offset_x * self.tiers
            shift_y += offset_y * self.tiers

            if self.mode == 1:
                cos_a = math.cos(-math.pi / 6)
                sin_a = math.sin(-math.pi / 6)
                final_x = shift_x * cos_a - shift_y * sin_a
                final_y = shift_x * sin_a + shift_y * cos_a
            else:
                final_x, final_y = shift_x, shift_y

            for x, y, z, tier in central_bs:
                all_bs.append((x + final_x, y + final_y, z, tier, wrap_id))

        return all_bs

    def generate_user_coordinates(self) -> List[Tuple[float, float, float]]:
        if self.total_users <= 0:
            return []

        hex_coords = self.generate_bs_coordinates()
        if len(hex_coords) == 0:
            return []

        user_coords = []
        for _ in range(self.total_users):
            hex_index = random.randint(0, len(hex_coords) - 1)
            hex_x, hex_y, hex_z, _ = hex_coords[hex_index]
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
            user_z = random.uniform(0, self.bs_height)

            user_coords.append((user_x, user_y, user_z))

        return user_coords

    def get_bs_user_length(self, x_user: float, y_user: float, z_user: float, x_bs: float, y_bs: float, z_bs: float) -> float:
        return math.sqrt((x_bs - x_user) ** 2 + (y_bs - y_user) ** 2 + (z_bs - z_user) ** 2)

    def save_distances_to_csv(self, filename: str = "user_bs_distances.csv",user_coords: List = None,use_wrap_around: bool = False):
        if user_coords is None:
            user_coords = self.generate_user_coordinates()

        if use_wrap_around:
            bs_coords = self.generate_wrapped_bs_positions()
        else:
            bs_coords = self.get_bs_positions()

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['bs_coordinates', 'user_coordinates', 'distance_m'])

            for user_x, user_y, user_z in user_coords:
                min_dist = float('inf')
                nearest_bs = (0.0, 0.0, 0.0)
                for bs_x, bs_y, bs_z, _, _ in bs_coords:
                    dist = self.get_bs_user_length(user_x, user_y, user_z, bs_x, bs_y, bs_z)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_bs = (bs_x, bs_y, bs_z)

                bs_coord_str = f"({nearest_bs[0]:.2f}, {nearest_bs[1]:.2f}, {nearest_bs[2]:.2f})"
                user_coord_str = f"({user_x:.2f}, {user_y:.2f}, {user_z:.2f})"
                writer.writerow([bs_coord_str, user_coord_str, f"{min_dist:.2f}"])
        print(f"Сохранено {len(user_coords) * len(bs_coords)} записей в {filename}")
        print(f"Пользователей: {len(user_coords)}")
        print(f"БС (с копиями): {len(bs_coords)}")

    def visualize(self, user_coords: List = None):
        bs_coords = self.generate_wrapped_bs_positions()
        if user_coords is None:
            user_coords = self.generate_user_coordinates()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        if user_coords:
            user_x = [x for x, y, z in user_coords]
            user_y = [y for x, y, z in user_coords]
            ax.scatter(user_x, user_y, c='red', s=30, alpha=0.7, marker='o', zorder=5)

        for x, y, z, tier, wrap_id in bs_coords:
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=self.cell_radius,
                orientation=0,
                edgecolor='black',
                alpha=0.3 if wrap_id > 0 else 0.5,
                linewidth=0.8 if wrap_id > 0 else 1.2,
                linestyle='--' if wrap_id > 0 else '-'
            )
            ax.add_patch(hexagon)

        text_offset = self.ISD * 0.1
        for x, y, z, tier, wrap_id in bs_coords:
            if wrap_id == 0:
                ax.plot(x, y, 'ko', markersize = 8)
                ax.text(x, y + text_offset, f'({x:.0f}, {y:.0f})',fontsize = 8, color='black')
            else:
                ax.plot(x, y, 'go', markersize = 6  , alpha=0.5)
                ax.text(x, y + text_offset, f'({x:.0f}, {y:.0f})',fontsize = 6, color='green', alpha=0.7)

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X (м)', fontsize =  12)
        ax.set_ylabel('Y (м)', fontsize =  12)

        ax.set_title(f'Гексагональная сеть: ISD={self.ISD} м', fontsize = 14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    network = HexagonalNetwork("config.json")
    user_coords = network.generate_user_coordinates()

    network.save_distances_to_csv("user_bs_distances.csv", user_coords=user_coords, use_wrap_around=True)

    network.visualize(user_coords=user_coords)

if __name__ == "__main__":
    main()