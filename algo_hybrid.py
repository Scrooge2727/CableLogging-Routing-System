import numpy as np
import random
import heapq
from shapely.geometry import Polygon, Point, LineString
import config


class FastPathPlanner:
    def __init__(self, start, targets, poly, obstacles, max_segment):
        self.start = start
        self.targets = targets
        self.poly = poly
        self.obstacles = obstacles
        self.max_segment = max_segment

        self.nodes = [start] + targets + list(poly.exterior.coords)[:-1]

        for ox, oy, r in obstacles:
            for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                px, py = ox + (r + 0.1) * np.cos(angle), oy + (r + 0.1) * np.sin(angle)
                if poly.contains(Point(px, py)):
                    self.nodes.append((px, py))

        self.graph = self._build_visibility_graph()

    def _build_visibility_graph(self):
        graph = {i: [] for i in range(len(self.nodes))}
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = np.hypot(self.nodes[i][0] - self.nodes[j][0], self.nodes[i][1] - self.nodes[j][1])

                # ОГРАНИЧЕНИЕ: Если расстояние больше max_segment, ребро даже не создается!
                if dist <= self.max_segment:
                    line = LineString([self.nodes[i], self.nodes[j]])
                    if (self.poly.contains(line) or self.poly.covers(line)) and not any(
                            line.distance(Point(ox, oy)) < r - 1e-3 for ox, oy, r in self.obstacles):
                        graph[i].append((j, dist))
                        graph[j].append((i, dist))
        return graph

    def dijkstra(self, start_idx, goal_idx):
        queue = [(0, start_idx, [start_idx])]
        visited = set()
        while queue:
            cost, current, path = heapq.heappop(queue)
            if current == goal_idx: return path
            if current in visited: continue
            visited.add(current)
            for neighbor, weight in self.graph[current]:
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
        return None

    def get_paths(self):
        paths = []
        for i, target in enumerate(self.targets):
            target_idx = i + 1
            path_indices = self.dijkstra(0, target_idx)
            if path_indices:
                paths.append([self.nodes[idx] for idx in path_indices])
        return paths


def run_hybrid(scenario, l_max, generations, max_segment):
    poly_points = scenario["polygon"]
    poly = Polygon(poly_points)
    start_point = scenario["start"]
    obstacles = scenario["obstacles"]

    NUM_TARGETS = 7
    POPULATION_SIZE = 50

    minx, miny, maxx, maxy = poly.bounds
    grid_points = config.generate_grid(poly_points, obstacles)

    def get_random_valid_point():
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p) and all(p.distance(Point(ox, oy)) > r for ox, oy, r in obstacles):
                return (p.x, p.y)

    def get_fitness(chromosome):
        targets = [(chromosome[i], chromosome[i + 1]) for i in range(0, len(chromosome), 2)]

        # Передаем max_segment в планировщик графа
        planner = FastPathPlanner(start_point, targets, poly, obstacles, max_segment)
        paths = planner.get_paths()

        if not paths: return -100000, []

        lines = [LineString(p) for p in paths]
        covered = sum(1 for pt in grid_points if any(line.distance(pt) <= l_max for line in lines))
        coverage_pct = (covered / len(grid_points)) * 100 if len(grid_points) > 0 else 0

        total_len = sum(line.length for line in lines)
        total_supports = sum(len(p) - 2 for p in paths)

        coverage_score = coverage_pct * 1000
        len_penalty = total_len * 1.0
        support_penalty = total_supports * 20.0

        # Если цель не была достигнута (т.к. точки графа были слишком далеко друг от друга), штрафуем
        unreachable_penalty = (NUM_TARGETS - len(paths)) * 5000

        fitness = coverage_score - len_penalty - support_penalty - unreachable_penalty
        return fitness, lines

    def mutate(chrom):
        new_chrom = chrom[:]
        if random.random() < 0.5:
            idx = random.randint(0, NUM_TARGETS - 1) * 2
            nx, ny = get_random_valid_point()
            new_chrom[idx], new_chrom[idx + 1] = nx, ny
        return new_chrom

    population = []
    for _ in range(POPULATION_SIZE):
        chrom = []
        for _ in range(NUM_TARGETS):
            px, py = get_random_valid_point()
            chrom.extend([px, py])
        population.append(chrom)

    for gen in range(generations):
        pop_eval = sorted([(c, *get_fitness(c)) for c in population], key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in pop_eval[:10]]
        while len(next_gen) < POPULATION_SIZE:
            tourn = random.sample(pop_eval[:20], 3)
            p1 = max(tourn, key=lambda x: x[1])[0]
            next_gen.append(mutate(p1))
        population = next_gen

    return pop_eval[0][2]