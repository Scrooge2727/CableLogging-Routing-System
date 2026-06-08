import numpy as np
import random
from shapely.geometry import Polygon, Point, LineString
import config


def segment_circle_intersection(p1, p2, c, r):
    d = np.array(p2) - np.array(p1)
    A = np.dot(d, d)
    if A == 0: return None
    B = 2 * np.dot(d, np.array(p1) - np.array(c))
    C = np.dot(np.array(p1) - np.array(c), np.array(p1) - np.array(c)) - r * r
    D = B * B - 4 * A * C
    if D < 0: return None
    ts = [t for t in ((-B - np.sqrt(D)) / (2 * A), (-B + np.sqrt(D)) / (2 * A)) if 0 < t < 1]
    return min(ts) if ts else None


def run_pure_ga(scenario, l_max, generations, max_segment):
    poly = Polygon(scenario["polygon"])
    start_point = Point(scenario["start"])
    obstacles = scenario["obstacles"]

    NUM_RAYS = 6
    POPULATION_SIZE = 50

    grid_points = config.generate_grid(scenario["polygon"], scenario["obstacles"])

    minx, miny, maxx, maxy = poly.bounds
    max_diagonal = np.hypot(maxx - minx, maxy - miny)

    def raycast(p_start, p_target):
        line = LineString([p_start, p_target])

        # Если целевая точка слишком близко, нет смысла считать
        if line.length < 1e-6:
            return p_start

        closest_pt = p_target

        # Используем covers, так как он надежнее работает с границами
        if not poly.covers(line):
            inter = line.intersection(poly.exterior)

            if inter.is_empty:
                # Линия не в полигоне и не пересекает границу -> она полностью снаружи
                closest_pt = p_start
            else:
                pts = []
                if inter.geom_type == 'Point':
                    pts.append(inter)
                elif inter.geom_type in ['MultiPoint', 'GeometryCollection']:
                    for geom in getattr(inter, 'geoms', [inter]):
                        if geom.geom_type == 'Point':
                            pts.append(geom)
                        elif geom.geom_type == 'LineString':
                            pts.append(Point(geom.coords[0]))
                            pts.append(Point(geom.coords[-1]))
                elif inter.geom_type == 'LineString':
                    pts.append(Point(inter.coords[0]))
                    pts.append(Point(inter.coords[-1]))

                # ВАЖНО: Игнорируем саму стартовую точку (погрешность 1e-6)
                valid_pts = [p for p in pts if p.distance(p_start) > 1e-6]

                if valid_pts:
                    # Находим ближайшую реальную точку пересечения
                    closest_pt = min(valid_pts, key=lambda p: p.distance(p_start))
                else:
                    # Если все точки пересечения - это старт, значит мы уперлись в границу и пытаемся выйти
                    closest_pt = p_start

        # Проверка препятствий остается без изменений
        start_np = np.array([p_start.x, p_start.y])
        end_np = np.array([closest_pt.x, closest_pt.y])

        for ox, oy, r in obstacles:
            t = segment_circle_intersection(start_np, end_np, (ox, oy), r)
            if t is not None:
                end_np = start_np + (end_np - start_np) * t

        return Point(end_np[0], end_np[1])

    def split_into_segments(p_start, p_end):
        dist = p_start.distance(p_end)
        if dist <= 0.1: return []

        pts = [p_start]
        current_dist = max_segment

        dx = (p_end.x - p_start.x) / dist
        dy = (p_end.y - p_start.y) / dist

        while current_dist < dist:
            pts.append(Point(p_start.x + dx * current_dist, p_start.y + dy * current_dist))
            current_dist += max_segment

        pts.append(p_end)
        return pts

    def decode_and_repair(chromosome):
        lines = []

        for a1, l1, delta_a, l2 in chromosome:
            # 1. Первая прямая (до поворотного башмака)
            p1_raw = Point(start_point.x + l1 * np.cos(np.radians(a1)), start_point.y + l1 * np.sin(np.radians(a1)))
            p1 = raycast(start_point, p1_raw)

            # 2. Вторая прямая (после излома)
            a2 = a1 + delta_a
            p2_raw = Point(p1.x + l2 * np.cos(np.radians(a2)), p1.y + l2 * np.sin(np.radians(a2)))
            p2 = raycast(p1, p2_raw)

            pts = []

            # ЗАЩИТА ОТ "ПЕНЬКОВ": Игнорируем отрезки короче 2 метров
            if start_point.distance(p1) > 2.0:
                pts1 = split_into_segments(start_point, p1)
                pts.extend(pts1)

                if p1.distance(p2) > 2.0:
                    pts2 = split_into_segments(p1, p2)
                    pts.extend(pts2[1:])

            if len(pts) > 1:
                lines.append(LineString([(p.x, p.y) for p in pts]))

        return lines

    def get_fitness(chromosome):
        lines = decode_and_repair(chromosome)
        if not lines: return -100000

        covered = sum(1 for pt in grid_points if any(line.distance(pt) <= l_max for line in lines))
        cov_pct = (covered / len(grid_points)) * 100 if len(grid_points) > 0 else 0

        total_len = sum(line.length for line in lines)

        # Оставляем только штраф за "паутину" (пересечение тросов друг с другом)
        intersections = 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i].crosses(lines[j]): intersections += 1
        intersect_penalty = intersections * 50000

        # УБРАЛИ штраф за острые углы. Теперь алгоритм может гнуть трос как угодно ради покрытия!
        return (cov_pct * 1000) - (total_len * 2.0) - intersect_penalty

    def mutate(chrom):
        new_chrom = [g[:] for g in chrom]
        if random.random() < 0.5:
            idx, p = random.randint(0, NUM_RAYS - 1), random.randint(0, 3)
            if p == 0:
                new_chrom[idx][p] += random.uniform(-25, 25)
            elif p == 2:
                new_chrom[idx][p] += random.uniform(-45, 45)
                new_chrom[idx][p] = max(-180, min(180, new_chrom[idx][p]))
            else:
                new_chrom[idx][p] += random.uniform(-5, 5)
                new_chrom[idx][p] = max(0.5, min(max_diagonal, new_chrom[idx][p]))
        return new_chrom

    # Инициализация длин и свободных углов (от -180 до 180)
    population = [
        [[random.uniform(0, 360), random.uniform(1, max_diagonal),
          random.uniform(-180, 180), random.uniform(1, max_diagonal)] for _ in
         range(NUM_RAYS)] for _ in range(POPULATION_SIZE)]

    for gen in range(generations):
        pop_eval = sorted([(c, get_fitness(c)) for c in population], key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in pop_eval[:10]]
        while len(next_gen) < POPULATION_SIZE:
            p1, p2 = random.choice(pop_eval[:20])[0], random.choice(pop_eval[:20])[0]
            cut = random.randint(1, NUM_RAYS - 1)
            next_gen.append(mutate(p1[:cut] + p2[cut:]))
        population = next_gen

    final_lines = decode_and_repair(population[0])
    return final_lines