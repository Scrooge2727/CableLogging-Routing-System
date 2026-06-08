import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import json
import time
from datetime import datetime

import config
from algo_fan import run_fan
from algo_pure_ga import run_pure_ga
from algo_hybrid import run_hybrid


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ИС 'Оптимизация лесозаготовок'")
        self.root.geometry("1250x800")

        left_frame = tk.Frame(root, width=320, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # --- 1. БЛОК ВЫБОРА И ЗАГРУЗКИ ---
        tk.Label(left_frame, text="1. Выберите или загрузите лесосеку:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(
            pady=(0, 5))

        self.combo_scenario = ttk.Combobox(left_frame, values=list(config.SCENARIOS.keys()), width=35)
        self.combo_scenario.current(0)
        self.combo_scenario.pack(pady=5)
        self.combo_scenario.bind("<<ComboboxSelected>>", self.draw_initial)

        tk.Button(left_frame, text="Загрузить из .JSON файла", command=self.load_custom_json, width=30,
                  bg="#e0e0e0").pack(pady=2)
        tk.Button(left_frame, text="Показать исходные данные", command=self.draw_initial, width=30).pack(pady=5)

        # --- 2. БЛОК ПАРАМЕТРОВ ---
        tk.Label(left_frame, text="2. Настройка параметров:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(
            pady=(15, 5))

        tk.Label(left_frame, text="Покрытие дерева (Эпсилон, м):", bg="#f0f0f0").pack()
        self.scale_lmax = tk.Scale(left_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=200,
                                   bg="#f0f0f0")
        self.scale_lmax.set(1.3)
        self.scale_lmax.pack()

        tk.Label(left_frame, text="Макс. пролет между опорами (м):", bg="#f0f0f0").pack()
        self.scale_max_seg = tk.Scale(left_frame, from_=5.0, to=50.0, resolution=1.0, orient=tk.HORIZONTAL, length=200,
                                      bg="#f0f0f0")
        self.scale_max_seg.set(20.0)
        self.scale_max_seg.pack()

        tk.Label(left_frame, text="Количество поколений ГА:", bg="#f0f0f0").pack()
        self.scale_gen = tk.Scale(left_frame, from_=10, to=150, resolution=10, orient=tk.HORIZONTAL, length=200,
                                  bg="#f0f0f0")
        self.scale_gen.set(50)
        self.scale_gen.pack()

        # --- 3. БЛОК ЗАПУСКА ---
        tk.Label(left_frame, text="3. Запуск алгоритмов:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        tk.Button(left_frame, text="Веерный алгоритм (Базовый)", command=self.process_fan, width=30, bg="#d9edf7").pack(
            pady=3)
        tk.Button(left_frame, text="Полный ГА (Прямой синтез)", command=self.process_pure_ga, width=30,
                  bg="#dff0d8").pack(pady=3)
        tk.Button(left_frame, text="Гибрид (ГА + Дейкстра)", command=self.process_hybrid, width=30, bg="#fcf8e3").pack(
            pady=3)

        tk.Frame(left_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)
        tk.Button(left_frame, text="ЭКСПОРТ (Прогнать все и выгрузить)", command=self.run_all_and_save, width=30,
                  bg="#ffcccb", font=("Arial", 9, "bold")).pack(pady=5)
        tk.Button(left_frame, text="СОБРАТЬ СТАТИСТИКУ (30 запусков)", command=self.collect_statistics, width=30,
                  bg="#e2c4ff", font=("Arial", 9, "bold")).pack(pady=5)

        self.lbl_status = tk.Label(left_frame, text="Статус: Ожидание", fg="blue", bg="#f0f0f0", font=("Arial", 10))
        self.lbl_status.pack(side=tk.BOTTOM, pady=20)

        # --- ПРАВАЯ ПАНЕЛЬ ---
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_initial()

    def set_status(self, text, color="blue"):
        self.lbl_status.config(text=f"Статус: {text}", fg=color)
        self.root.update()

    def get_params(self):
        return {
            "scenario": config.SCENARIOS[self.combo_scenario.get()],
            "l_max": self.scale_lmax.get(),
            "max_seg": self.scale_max_seg.get(),
            "generations": self.scale_gen.get()
        }

    # ==========================================
    # БЛОК 1: ВАЛИДАЦИЯ ГЕОМЕТРИИ (При загрузке)
    # ==========================================
    def validate_geometry(self, scenario):
        from shapely.geometry import Polygon, Point
        try:
            poly = Polygon(scenario["polygon"])
            start = Point(scenario["start"])
            obstacles = scenario["obstacles"]

            if not poly.is_valid:
                return False, "Топология лесосеки некорректна (присутствуют самопересечения границ полигона)."

            for idx, (ox, oy, r) in enumerate(obstacles):
                if r <= 0:
                    return False, f"Радиус препятствия #{idx + 1} должен быть строго больше нуля."

            if not (poly.contains(start) or poly.touches(start)):
                return False, "Ошибка: Корневая точка находится за физическими границами лесосеки."

            for idx, (ox, oy, r) in enumerate(obstacles):
                if start.distance(Point(ox, oy)) < (r - 1e-6):
                    return False, f"Ошибка: Корневая точка попадает внутрь препятствия #{idx + 1}."
        except Exception as e:
            return False, f"Критическая ошибка геометрии JSON файла: {str(e)}"

        return True, "OK"

    # ==========================================
    # БЛОК 2: ВАЛИДАЦИЯ ПАРАМЕТРОВ (При запуске)
    # ==========================================
    def validate_parameters(self, params):
        if params["l_max"] <= 0 or params["max_seg"] <= 0:
            return False, "Технологические параметры (радиус покрытия, максимальный пролет) должны быть положительными числами."
        if params["max_seg"] < params["l_max"]:
            return False, "Логическая ошибка: Максимальная длина пролета каната не может быть меньше радиуса зоны обслуживания."
        return True, "OK"

    # НОВАЯ ФУНКЦИЯ: Парсинг JSON с мгновенной валидацией
    def load_custom_json(self):
        filepath = filedialog.askopenfilename(
            title="Выберите файл лесосеки",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            name = data.get("name", "Custom Scenario")
            polygon = [tuple(p) for p in data["polygon"]]
            start = tuple(data["start"])
            obstacles = [tuple(o) for o in data["obstacles"]]

            # Формируем кандидата на добавление
            scenario_candidate = {
                "polygon": polygon,
                "start": start,
                "obstacles": obstacles
            }

            # --- ВАЛИДАЦИЯ В МОМЕНТ ЗАГРУЗКИ ---
            is_valid, msg = self.validate_geometry(scenario_candidate)
            if not is_valid:
                messagebox.showerror("Ошибка валидации файла", f"Файл отклонен системой валидации:\n\n{msg}")
                self.set_status("Отклонено: кривой JSON", "red")
                return
            # -----------------------------------

            config.SCENARIOS[name] = scenario_candidate
            self.combo_scenario['values'] = list(config.SCENARIOS.keys())
            self.combo_scenario.set(name)
            self.draw_initial()
            self.set_status("Файл проверен и загружен!", "green")

        except Exception as e:
            messagebox.showerror("Ошибка чтения", f"Файл поврежден или имеет неверную структуру JSON.\n\n{e}")

    def draw_canvas(self, params, lines=None, title="Исходная лесосека"):
        self.ax.clear()
        scenario, l_max = params["scenario"], params["l_max"]

        poly_pts = scenario["polygon"].copy()
        poly_pts.append(poly_pts[0])


        self.ax.plot(*zip(*poly_pts), color='darkgreen', linestyle='-', lw=2.5, label='Граница леса')
        # ------------------------------------------

        for ox, oy, r in scenario["obstacles"]:
            self.ax.add_patch(plt.Circle((ox, oy), r, color='orange', alpha=0.5))

        if scenario["obstacles"]:
            self.ax.scatter([], [], s=80, c='orange', alpha=0.5, label='Препятствие')

        start = scenario["start"]
        self.ax.plot(start[0], start[1], 'r^', markersize=12, label='Корневая точка', zorder=5)

        if lines:
            grid = config.generate_grid(scenario["polygon"], scenario["obstacles"])
            pts = np.array([(p.x, p.y) for p in grid])

            mask = np.array([any(line.distance(p) <= l_max for line in lines) for p in grid])
            coverage = (mask.sum() / len(grid)) * 100 if len(grid) > 0 else 0

            self.ax.scatter(pts[mask, 0], pts[mask, 1], s=6, c='blue', alpha=0.5, label='Покрытая зона')
            self.ax.scatter(pts[~mask, 0], pts[~mask, 1], s=6, c='red', label='Непокрытая зона')

            for idx, line in enumerate(lines):
                self.ax.plot(*line.xy, 'k-', lw=2, zorder=4, label='Трос' if idx == 0 else "")
                self.ax.plot(*line.xy, 'ko', mfc='white', zorder=5, label='Опора' if idx == 0 else "")

            title += f"\nПокрытие: {coverage:.1f}% | Расход троса: {sum(l.length for l in lines):.1f}м"

        self.ax.set_aspect('equal')
        self.ax.set_title(title)

        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')

        # Подгоняем размеры холста, чтобы вынесенная легенда не обрезалась краем окна
        self.fig.tight_layout()
        # -----------------------------------------------------

        self.canvas.draw()

    def draw_initial(self, event=None):
        self.set_status("Ожидание")
        self.draw_canvas(self.get_params())

    def process_fan(self):
        p = self.get_params()
        if not self.validate_parameters(p)[0]:
            messagebox.showwarning("Ошибка параметров", self.validate_parameters(p)[1])
            return None

        self.set_status("Считаю Веер...", "red")
        lines = run_fan(p["scenario"], p["l_max"], p["max_seg"])
        self.draw_canvas(p, lines, f"Веерный алгоритм")
        self.set_status("Веер построен", "green")
        return lines

    def process_pure_ga(self):
        p = self.get_params()
        if not self.validate_parameters(p)[0]:
            messagebox.showwarning("Ошибка параметров", self.validate_parameters(p)[1])
            return None

        self.set_status(f"Считаю Полный ГА...", "red")
        lines = run_pure_ga(p["scenario"], p["l_max"], p["generations"], p["max_seg"])
        self.draw_canvas(p, lines, f"Полный ГА (Прямой синтез)")
        self.set_status("Полный ГА построен", "green")
        return lines

    def process_hybrid(self):
        p = self.get_params()
        if not self.validate_parameters(p)[0]:
            messagebox.showwarning("Ошибка параметров", self.validate_parameters(p)[1])
            return None

        self.set_status(f"Считаю Гибрид...", "red")
        lines = run_hybrid(p["scenario"], p["l_max"], p["generations"], p["max_seg"])
        self.draw_canvas(p, lines, f"Гибрид (ГА + Дейкстра)")
        self.set_status("Гибрид построен", "green")
        return lines

    def evaluate_metrics(self, lines, scenario, l_max):
        MAX_TARGETS = 7
        if not lines: return {"coverage": 0, "length": 0, "supports": 0, "intersections": 0, "unreachable": MAX_TARGETS}

        grid_points = config.generate_grid(scenario["polygon"], scenario["obstacles"])

        cov = sum(1 for pt in grid_points if any(line.distance(pt) <= l_max for line in lines)) / len(
            grid_points) * 100 if grid_points else 0
        l_tot = sum(line.length for line in lines)
        k_sup = sum(max(0, len(line.coords) - 2) for line in lines)
        cross = sum(1 for i in range(len(lines)) for j in range(i + 1, len(lines)) if lines[i].crosses(lines[j]))
        unr = max(0, MAX_TARGETS - len(lines))

        return {"coverage": cov, "length": l_tot, "supports": k_sup, "intersections": cross, "unreachable": unr}

    def save_algorithm_results_json(self, base_folder, algo_name, lines, metrics, scenario_name):
        if not lines: return
        algo_dir = os.path.join(base_folder, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        self.fig.savefig(os.path.join(algo_dir, f"{algo_name}_plot.png"), dpi=300, bbox_inches='tight')

        serializable_lines = [[list(pt) for pt in line.coords] for line in lines]

        export_data = {
            "metadata": {"scenario": scenario_name, "algorithm": algo_name,
                         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "metrics": {
                "coverage_pct": round(metrics.get("coverage", 0.0), 2),
                "total_length_m": round(metrics.get("length", 0.0), 2),
                "total_supports": metrics.get("supports", 0),
                "intersections": metrics.get("intersections", 0),
                "unreachable_targets": metrics.get("unreachable", 0),
                "execution_time_s": round(metrics.get("time", 0.0), 2)
            },
            "geometry": {"routes": serializable_lines}
        }
        with open(os.path.join(algo_dir, f"{algo_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)

    def run_all_and_save(self):
        p = self.get_params()
        if not self.validate_parameters(p)[0]:
            messagebox.showwarning("Пакетный запуск прерван", self.validate_parameters(p)[1])
            return

        scenario_name = self.combo_scenario.get().split()[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_folder = f"Эксперимент_{scenario_name}_{timestamp}"
        os.makedirs(base_folder, exist_ok=True)

        # Веер
        self.set_status("Считаю: Веер...", "red")
        t_start = time.time()
        lines_fan = self.process_fan()
        metrics_fan = self.evaluate_metrics(lines_fan, p["scenario"], p["l_max"])
        metrics_fan["time"] = time.time() - t_start
        self.save_algorithm_results_json(base_folder, "1_Fan", lines_fan, metrics_fan, scenario_name)

        # Прямой ГА
        self.set_status("Считаю: Полный ГА...", "red")
        t_start = time.time()
        lines_ga = self.process_pure_ga()
        metrics_ga = self.evaluate_metrics(lines_ga, p["scenario"], p["l_max"])
        metrics_ga["time"] = time.time() - t_start
        self.save_algorithm_results_json(base_folder, "2_PureGA", lines_ga, metrics_ga, scenario_name)

        # Гибрид
        self.set_status("Считаю: Гибрид...", "red")
        t_start = time.time()
        lines_hybrid = self.process_hybrid()
        metrics_hybrid = self.evaluate_metrics(lines_hybrid, p["scenario"], p["l_max"])
        metrics_hybrid["time"] = time.time() - t_start
        self.save_algorithm_results_json(base_folder, "3_Hybrid", lines_hybrid, metrics_hybrid, scenario_name)

        self.set_status(f"Успех! Данные сохранены в {base_folder}", "green")
        messagebox.showinfo("Экспорт завершен", f"Графики и JSON-отчеты сохранены в папку:\n{base_folder}")

    def collect_statistics(self):
        p = self.get_params()
        if not self.validate_parameters(p)[0]:
            messagebox.showwarning("Сбор статистики прерван", self.validate_parameters(p)[1])
            return

        scenario_name = self.combo_scenario.get().split()[0]
        scenario, l_max, max_seg, generations = p["scenario"], p["l_max"], p["max_seg"], p["generations"]
        N_RUNS = 30
        import random
        import numpy as np
        random.seed(67)
        np.random.seed(67)
        algos = [
            ("Веер (Базовый)", lambda: run_fan(scenario, l_max, max_seg)),
            ("Прямой ГА", lambda: run_pure_ga(scenario, l_max, generations, max_seg)),
            ("Гибрид", lambda: run_hybrid(scenario, l_max, generations, max_seg))
        ]

        filename = f"Статистика_{scenario_name}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== Сценарий: {scenario_name} | Запусков: {N_RUNS} ===\n")
            f.write(f"Параметры: l_max={l_max}, L_span={max_seg}, Gen={generations}\n\n")

            for algo_name, algo_func in algos:
                self.set_status(f"Сбор статистики: {algo_name} (0/{N_RUNS})...", "red")
                self.root.update()

                res_cov, res_len, res_sup, res_cross, res_unr, res_time = [], [], [], [], [], []

                for i in range(N_RUNS):
                    t_start = time.time()
                    lines = algo_func()
                    t_calc = time.time() - t_start

                    metrics = self.evaluate_metrics(lines, scenario, l_max)

                    res_cov.append(metrics["coverage"])
                    res_len.append(metrics["length"])
                    res_sup.append(metrics["supports"])
                    res_cross.append(metrics["intersections"])
                    res_unr.append(metrics["unreachable"])
                    res_time.append(t_calc)

                    self.set_status(f"Сбор статистики: {algo_name} ({i + 1}/{N_RUNS})...", "red")
                    self.root.update()

                f.write(f"--- {algo_name} ---\n")
                f.write(f"P_cov:   {np.mean(res_cov):.1f} ± {np.std(res_cov):.1f} %\n")
                f.write(f"L_tot:   {np.mean(res_len):.1f} ± {np.std(res_len):.1f} м\n")
                f.write(f"K_sup:   {np.mean(res_sup):.1f} ± {np.std(res_sup):.1f} шт\n")
                f.write(f"N_cross: {np.mean(res_cross):.1f} ± {np.std(res_cross):.1f} шт\n")
                f.write(f"N_unr:   {np.mean(res_unr):.1f} ± {np.std(res_unr):.1f} шт\n")
                f.write(f"t_calc:  {np.mean(res_time):.2f} ± {np.std(res_time):.2f} сек\n\n")

        self.set_status(f"Статистика собрана! Файл {filename}", "green")
        messagebox.showinfo("Готово", f"Статистика собрана и сохранена в файл:\n{filename}")

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        import sys
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()