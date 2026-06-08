import numpy as np
from shapely.geometry import LineString, Point
import config


def ray_segment_intersection(p, d, a, b):
    M = np.column_stack((d, -(np.array(b) - np.array(a))))
    if abs(np.linalg.det(M)) < 1e-9: return None
    t, u = np.linalg.solve(M, np.array(a) - np.array(p))
    if t > 1e-6 and 0 <= u <= 1: return t
    return None


def segment_circle_intersection(p1, p2, c, r):
    d = np.array(p2) - np.array(p1)
    A = np.dot(d, d)
    B = 2 * np.dot(d, np.array(p1) - np.array(c))
    C = np.dot(np.array(p1) - np.array(c), np.array(p1) - np.array(c)) - r * r
    D = B * B - 4 * A * C
    if D < 0: return None
    ts = [t for t in ((-B - np.sqrt(D)) / (2 * A), (-B + np.sqrt(D)) / (2 * A)) if 0 < t < 1]
    return min(ts) if ts else None


def run_fan(scenario, l_max, max_segment):
    poly = scenario["polygon"]
    start = scenario["start"]
    obstacles = scenario["obstacles"]

    num_virtual_rays = 72
    angles = np.linspace(-np.pi, np.pi, num_virtual_rays)
    candidates = []

    for ang in angles:
        d = np.array([np.cos(ang), np.sin(ang)])
        t_hits = []
        for i in range(len(poly)):
            t = ray_segment_intersection(start, d, poly[i], poly[(i + 1) % len(poly)])
            if t: t_hits.append(t)
        if not t_hits: continue

        t_end = min(t_hits)
        end = np.array(start) + d * t_end

        # Обрезаем об препятствия
        for ox, oy, r in obstacles:
            c = (ox, oy)
            t = segment_circle_intersection(start, end, c, r)
            if t: end = np.array(start) + (end - np.array(start)) * t

        dist = np.linalg.norm(end - np.array(start))
        if dist > 0.5:
            # РЕШЕНИЕ: Разбиваем длинный луч на отрезки, чтобы появились промежуточные опоры
            pts = [tuple(start)]
            current_dist = max_segment

            # Ставим опору каждые max_segment метров
            while current_dist < dist:
                intermediate_pt = np.array(start) + d * current_dist
                pts.append(tuple(intermediate_pt))
                current_dist += max_segment

            pts.append(tuple(end))  # Добавляем конец троса

            candidates.append(LineString(pts))

    grid_points = config.generate_grid(poly, obstacles)
    uncovered_pts = set(grid_points)
    selected_lines = []
    MAX_RAYS = 7

    for _ in range(MAX_RAYS):
        best_line = None
        best_cover_count = -1
        best_covered_set = set()

        for line in candidates:
            if line in selected_lines: continue
            covers = {pt for pt in uncovered_pts if line.distance(pt) <= l_max}
            if len(covers) > best_cover_count:
                best_cover_count = len(covers)
                best_line = line
                best_covered_set = covers

        if best_line and best_cover_count > 0:
            selected_lines.append(best_line)
            uncovered_pts -= best_covered_set
        else:
            break

    return selected_lines