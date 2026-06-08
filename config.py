from shapely.geometry import Polygon, Point
import numpy as np

# Словарь с предопределенными сценариями
SCENARIOS = {
    "Сложная лесосека (Пользовательская)": {
        "polygon": [(0, 0), (9, 2), (18, 0.4), (25, 3), (27, 9), (26, 17),
                    (22, 22), (14, 24), (6, 23), (10, 18), (2, 11), (1.2, 5)],
        "start": (1, 1),
        "obstacles": [
            (5.8, 10.8, 0.55), (10.8, 4.8, 0.55), (3.8, 4.8, 0.55),
            (6.5, 7.2, 0.45), (9.5, 10.0, 0.5), (12.0, 13.0, 0.55),
            (15.0, 16.0, 0.45), (18.0, 5.5, 0.5), (18.0, 18.5, 0.5),
            (20.5, 14.5, 0.45), (17.0, 8.0, 0.4)
        ]
    },
    "Лесосека 'Подкова' (Ловушка для веера)": {
        "polygon": [(0, 0), (20, 0), (25, 5), (25, 20), (0, 20), (0, 14), (17, 14), (17, 6), (0, 6)],
        "start": (1, 1),
        "obstacles": [(19, 10, 1.5), (22, 17, 1.0), (22, 3, 1.0)]
    },
    "Лесосека 'Архипелаг' (Много преград)": {
        "polygon": [(0, 0), (30, 0), (30, 20), (0, 20)],
        "start": (2, 10),
        "obstacles": [(8, 5, 2), (8, 15, 2), (15, 10, 2.5), (22, 5, 2), (22, 15, 2), (15, 3, 1), (15, 17, 1)]
    }
}

def generate_grid(poly_points, obstacles):
    poly = Polygon(poly_points)
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    for x in np.arange(minx, maxx, 0.5):
        for y in np.arange(miny, maxy, 0.5):
            p = Point(x, y)
            if poly.contains(p) and not any(p.distance(Point(ox, oy)) <= r for ox, oy, r in obstacles):
                pts.append(p)
    return pts