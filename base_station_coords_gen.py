import json
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import random
import numpy as np

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

        self._hex_centers_for_vis = []
        self._all_hex_centers_wrapped = []

    def load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def hex_round(self, x: float, y: float, z: float):
        l = round(x)
        m = round(y)
        n = round(z)

        s = l + m + n

        if s == 0:
            return l, m, n

        if s > 0:
            xr = x - l
            yr = y - m
            zr = z - n
        else:
            xr = l - x
            yr = m - y
            zr = n - z

        if xr <= yr and xr <= zr:
            l = - (m + n)
        elif yr <= xr and yr <= zr:
            m = - (l + n)
        else:
            n = - (l + m)

        return l, m, n

    def hex_to_cartesian(self, x: float, y: float, z: float) -> Tuple[float, float]:
        step = self.site_radius if self.mode == 1 else self.ISD
        X = (x + y / 2.0) * step
        Y = -(x + z) * (math.sqrt(3) / 2) * step
        return (X, Y)

    def add_hex_point(self, hex_coords: list, cartesian_coords: list, x: int, y: int, z: int, tier: int):
        hex_coords.append((x, y, z, tier))
        X, Y = self.hex_to_cartesian(x, y, z)
        Z = self.bs_height
        cartesian_coords.append((X, Y, Z, tier))

    def generate_hex_grid(self):
        hex_coords = [(0, 0, 0, 0)]
        cartesian_coords = [(0.0, 0.0, self.bs_height, 0)]

        for tier in range(1, self.tiers + 1):
            for side in range(6):
                for step in range(tier):
                    x = (tier - step) * self.HEX_DIRECTIONS[side][0] + step * self.HEX_DIRECTIONS[(side + 1) % 6][0]
                    y = (tier - step) * self.HEX_DIRECTIONS[side][1] + step * self.HEX_DIRECTIONS[(side + 1) % 6][1]
                    z = (tier - step) * self.HEX_DIRECTIONS[side][2] + step * self.HEX_DIRECTIONS[(side + 1) % 6][2]

                    self.add_hex_point(hex_coords, cartesian_coords, x, y, z, tier)
        return hex_coords, cartesian_coords

    def generate_bs_coordinates(self):
        if self.mode == 1:
            hex_coords, _ = self.generate_hex_grid()
            bs_coords, hex_centers = self.generate_tri_hex_centers(hex_coords)
            self._hex_centers_for_vis = hex_centers
            return bs_coords
        else:
            _, cartesian_coords = self.generate_hex_grid()
            self._hex_centers_for_vis = cartesian_coords
            return cartesian_coords

    def generate_tri_hex_centers(self,hex_coords: List[Tuple[int, int, int, int]]) -> Tuple[List[Tuple[float, float, float, int]],List[Tuple[float, float, float, int]]]:
        bs_coords = []
        hex_centers = []

        offsets = np.array([
            [2 / 3, -1 / 3, -1 / 3],
            [-1 / 3, 2 / 3, -1 / 3],
            [-1 / 3, -1 / 3, 2 / 3]
        ])

        rot_matrix = np.array([
            [(1 + math.sqrt(3)) / 3, 1 / 3, (1 - math.sqrt(3)) / 3],
            [(1 - math.sqrt(3)) / 3, (1 + math.sqrt(3)) / 3, 1 / 3],
            [1 / 3, (1 - math.sqrt(3)) / 3, (1 + math.sqrt(3)) / 3]
        ])

        for hx, hy, hz, tier in hex_coords:

            h = np.array([hx, hy, hz])
            r = np.sqrt(3) * (rot_matrix @ h)

            rx, ry, rz = self.hex_round(*r)

            bx, by = self.hex_to_cartesian(rx, ry, rz)
            bs_coords.append((bx, by, self.bs_height, tier))

            shifted = offsets + np.array([rx, ry, rz])

            for cx, cy, cz in shifted:
                hx_c, hy_c = self.hex_to_cartesian(cx, cy, cz)
                hex_centers.append((hx_c, hy_c, self.bs_height, tier))

        return bs_coords, hex_centers

    def get_bs_positions(self) -> List[Tuple[float, float, float, int]]:
        return self.generate_bs_coordinates()

    def generate_wrapped_bs_positions(self) -> List[Tuple[float, float, float, int, int, int]]:
        central_bs = self.get_bs_positions()
        all_bs = []

        self._all_hex_centers_wrapped = []

        for idx, (x, y, z, tier) in enumerate(central_bs):
            all_bs.append((x, y, z, tier, 0, idx))

        for hx, hy, hz, tier in self._hex_centers_for_vis:
            self._all_hex_centers_wrapped.append((hx, hy, tier, 0))

        wrap_directions = self.HEX_DIRECTIONS
        shift_distance = (2 * self.tiers + 1) * self.ISD

        for wrap_id, (dx, dy, dz) in enumerate(wrap_directions, start=1):
            shift_x = (dx + dy / 2) * shift_distance
            shift_y = (dy * math.sqrt(3) / 2) * shift_distance

            idx = (wrap_id + 1) % 6
            odx, ody, _ = self.HEX_DIRECTIONS[idx]
            offset_x = (odx + ody / 2) * self.ISD
            offset_y = (ody * math.sqrt(3) / 2) * self.ISD

            shift_x += offset_x * self.tiers
            shift_y += offset_y * self.tiers

            if self.mode == 1:
                cos_a = math.cos(-math.pi / 6)
                sin_a = math.sin(-math.pi / 6)
                final_x = shift_x * cos_a - shift_y * sin_a
                final_y = shift_x * sin_a + shift_y * cos_a
            else:
                final_x, final_y = shift_x, shift_y

            for orig_idx, (x, y, z, tier) in enumerate(central_bs):
                all_bs.append((x + final_x, y + final_y, z, tier, wrap_id, orig_idx))

            for hx, hy, hz, tier in self._hex_centers_for_vis:
                self._all_hex_centers_wrapped.append((hx + final_x, hy + final_y, tier, wrap_id))

        return all_bs

    def generate_user_coordinates(self) -> List[Tuple[float, float, float]]:
        if self.total_users <= 0:
            return []

        if self.mode == 0:
            cell_centers = self.generate_bs_coordinates()
        else:
            self.generate_bs_coordinates()
            cell_centers = self._hex_centers_for_vis

        if len(cell_centers) == 0:
            return []

        user_coords = []

        for _ in range(self.total_users):

            cx, cy, cz, _ = random.choice(cell_centers)

            sector = random.randint(0, 5)

            r1 = random.random()
            r2 = random.random()

            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2

            angle_offset = math.pi / 6

            angle_base = sector * math.pi / 3 + angle_offset
            angle_next = (sector + 1) * math.pi / 3 + angle_offset

            x2 = self.cell_radius * math.cos(angle_base)
            y2 = self.cell_radius * math.sin(angle_base)

            x3 = self.cell_radius * math.cos(angle_next)
            y3 = self.cell_radius * math.sin(angle_next)

            user_rel_x = r1 * x2 + r2 * x3
            user_rel_y = r1 * y2 + r2 * y3

            user_x = cx + user_rel_x
            user_y = cy + user_rel_y
            user_z = random.uniform(0, self.bs_height)

            user_coords.append((user_x, user_y, user_z))

        return user_coords

    def get_bs_user_length(self, x_user: float, y_user: float, z_user: float, x_bs: float, y_bs: float,z_bs: float) -> float:
        return math.sqrt((x_bs - x_user) ** 2 + (y_bs - y_user) ** 2 + (z_bs - z_user) ** 2)

    def save_distances_to_txt(self, filename: str = "user_bs_distances.txt", user_coords: List = None):
        if user_coords is None:
            user_coords = self.generate_user_coordinates()

        bs_coords = self.generate_wrapped_bs_positions()
        central_bs = self.get_bs_positions()
        num_orig_bs = len(central_bs)

        with open(filename, 'w', encoding='utf-8') as f:
            for uid, (ux, uy, uz) in enumerate(user_coords):
                best_matches = {i: {'dist': float('inf'), 'coords': (0.0, 0.0, 0.0), 'wrap_id': -1}
                                for i in range(num_orig_bs)}
                for bs in bs_coords:
                    bx, by, bz, _, wrap_id, orig_id = bs
                    dist = self.get_bs_user_length(ux, uy, uz, bx, by, bz)
                    if dist < best_matches[orig_id]['dist']:
                        best_matches[orig_id]['dist'] = dist
                        best_matches[orig_id]['coords'] = (bx, by, bz)
                        best_matches[orig_id]['wrap_id'] = wrap_id
                f.write(f"For user with coords ({ux:.2f}, {uy:.2f}, {uz:.2f}) distance to:\n")
                for i in range(num_orig_bs):
                    x, y, z = best_matches[i]['coords']
                    d = best_matches[i]['dist']
                    w_id = best_matches[i]['wrap_id']
                    f.write(f"bs{i} with coords ({x:.2f}, {y:.2f}, {z:.2f}) from {w_id} cluster, distance = {d:.2f}\n")
                f.write("\n")

        print(f"Сохранено {len(user_coords) * len(bs_coords)} записей в {filename}")
        print(f"Пользователей: {len(user_coords)}")
        print(f"БС (с копиями): {len(bs_coords)}")

    def visualize(self, user_coords: List = None):
        wrapped_bs = self.generate_wrapped_bs_positions()

        if user_coords is None:
            user_coords = self.generate_user_coordinates()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        for x, y, tier, wrap_id in self._all_hex_centers_wrapped:
            is_wrapped = wrap_id > 0
            hexagon = RegularPolygon(
                (x, y), numVertices=6, radius=self.cell_radius, orientation=0,
                edgecolor='gray' if is_wrapped else 'black',
                alpha=0.2 if is_wrapped else 0.5,
                linewidth=0.8 if is_wrapped else 1.2,
                linestyle='--' if is_wrapped else '-',
                zorder=1
            )
            ax.add_patch(hexagon)

        for x, y, z, tier, wrap_id, orig_id in wrapped_bs:
            if wrap_id == 0:
                ax.plot(x, y, 'ko', markersize=8, zorder=3)
                text_color = 'black'
            else:
                ax.plot(x, y, 'go', markersize=6, alpha=0.5, zorder=3)
                text_color = 'green'

            label = f"Cl {wrap_id}\nBS {orig_id}"
            ax.text(x, y + self.ISD * 0.15, label, fontsize=8, ha='center', va='bottom',
                    color=text_color, alpha=1, zorder=4)

        if user_coords:
            ax.scatter([u[0] for u in user_coords], [u[1] for u in user_coords],
                       c='red', s=30, alpha=0.7, marker='o', zorder=5)

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X (м)', fontsize=12)
        ax.set_ylabel('Y (м)', fontsize=12)
        ax.set_title(f'Гексагональная сеть: ISD={self.ISD} м', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    network = HexagonalNetwork("config.json")
    user_coords = network.generate_user_coordinates()
    network.save_distances_to_txt("user_bs_distances.txt", user_coords=user_coords)
    network.visualize(user_coords=user_coords)


if __name__ == "__main__":
    main()