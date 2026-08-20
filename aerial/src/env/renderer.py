"""PyGame visualization for the SACRED toy environment."""

from __future__ import annotations

from src.env.graph_env import EdgeId, GraphEnv, NodeId
from src.env.multi_agent import EpisodeMetrics


Color = tuple[int, int, int]


class PygameToyRenderer:
    """Live renderer for graph state, trucks, congestion, and episode metrics."""

    def __init__(
        self,
        width: int = 1180,
        height: int = 760,
        fps: int = 30,
        sim_ticks_per_second: float = 6.0,
    ) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("PyGame is required for live rendering. Install it with `pip install pygame`.") from exc

        self.pygame = pygame
        self.width = width
        self.height = height
        self.fps = fps
        self.margin = 70
        self.panel_width = 360
        self.running = True
        self.paused = False
        self.single_step_requested = False
        self.sim_ticks_per_second = sim_ticks_per_second
        self._last_step_time_ms = 0

        pygame.init()
        pygame.display.set_caption("SACRED Toy SDVRP")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 14)
        self.tiny_font = pygame.font.SysFont("Arial", 12)

    def should_advance(self) -> bool:
        """Return true when the next simulation tick should be executed."""

        pygame = self.pygame
        if self.single_step_requested:
            self.single_step_requested = False
            self._last_step_time_ms = pygame.time.get_ticks()
            return True
        if self.paused:
            return False

        now = pygame.time.get_ticks()
        interval_ms = 1000.0 / max(0.25, self.sim_ticks_per_second)
        if now - self._last_step_time_ms >= interval_ms:
            self._last_step_time_ms = now
            return True
        return False

    def render(
        self,
        env: GraphEnv,
        metrics: EpisodeMetrics,
        *,
        protagonist_reward: float = 0.0,
        antagonist_reward: float = 0.0,
        antagonist_action: dict[EdgeId, float] | None = None,
        feed: list[tuple[str, str]] | None = None,
    ) -> bool:
        """Draw one frame and return false if the window was closed."""

        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

        self.screen.fill((248, 249, 250))
        positions = self._screen_positions(env)
        self._draw_edges(env, positions)
        self._draw_routes(env, positions)
        self._draw_nodes(env, positions)
        self._draw_trucks(env, positions)
        self._draw_panel(env, metrics, protagonist_reward, antagonist_reward, antagonist_action or {}, feed or [])

        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.running

    def close(self) -> None:
        self.pygame.quit()

    def _handle_keydown(self, key: int) -> None:
        pygame = self.pygame
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_UP):
            self.sim_ticks_per_second = min(60.0, self.sim_ticks_per_second * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_DOWN):
            self.sim_ticks_per_second = max(0.25, self.sim_ticks_per_second / 1.5)
        elif key == pygame.K_RIGHT:
            self.single_step_requested = True
            self.paused = True

    def _screen_positions(self, env: GraphEnv) -> dict[NodeId, tuple[int, int]]:
        xs = [data["x"] for _, data in env.graph.nodes(data=True)]
        ys = [data["y"] for _, data in env.graph.nodes(data=True)]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        drawable_width = self.width - self.panel_width - (2 * self.margin)
        drawable_height = self.height - (2 * self.margin)

        def scale(value: float, lower: float, upper: float, size: int) -> int:
            if abs(upper - lower) < 1e-12:
                return self.margin + (size // 2)
            return self.margin + int(((value - lower) / (upper - lower)) * size)

        positions = {}
        for node, data in env.graph.nodes(data=True):
            x = scale(data["x"], min_x, max_x, drawable_width)
            y = self.height - scale(data["y"], min_y, max_y, drawable_height)
            positions[node] = (x, y)
        return positions

    def _draw_edges(self, env: GraphEnv, positions: dict[NodeId, tuple[int, int]]) -> None:
        pygame = self.pygame
        for u, v, data in env.graph.edges(data=True):
            congestion = data["congestion_level"]
            color = self._congestion_color(congestion)
            width = 3 + int(5 * congestion)
            pygame.draw.line(self.screen, color, positions[u], positions[v], width)

            midpoint = ((positions[u][0] + positions[v][0]) // 2, (positions[u][1] + positions[v][1]) // 2)
            if congestion > 0:
                label = self.small_font.render(f"{congestion:.2f}", True, (119, 29, 29))
                self.screen.blit(label, (midpoint[0] + 6, midpoint[1] - 18))

    def _draw_routes(self, env: GraphEnv, positions: dict[NodeId, tuple[int, int]]) -> None:
        pygame = self.pygame
        for truck in env.trucks.values():
            if len(truck.path) < 2 or truck.destination is None:
                continue
            for start, end in zip(truck.path[truck.path_index :], truck.path[truck.path_index + 1 :]):
                pygame.draw.line(self.screen, (71, 118, 230), positions[start], positions[end], 1)

    def _draw_nodes(self, env: GraphEnv, positions: dict[NodeId, tuple[int, int]]) -> None:
        pygame = self.pygame
        for node, data in env.graph.nodes(data=True):
            x, y = positions[node]
            if data["has_depot"]:
                color = (34, 139, 89)
                radius = 14
            elif data["demand"] > 0:
                color = (222, 116, 54)
                radius = 10 + int(2 * data["demand"])
            else:
                color = (130, 140, 150)
                radius = 8

            pygame.draw.circle(self.screen, color, (x, y), radius)
            pygame.draw.circle(self.screen, (31, 41, 55), (x, y), radius, 2)
            label = self.small_font.render(str(node), True, (20, 24, 31))
            self.screen.blit(label, (x + radius + 4, y - radius))
            if data["demand"] > 0:
                demand_label = self.small_font.render(f"d={data['demand']:.0f}", True, (20, 24, 31))
                self.screen.blit(demand_label, (x + radius + 4, y + 2))

    def _draw_trucks(self, env: GraphEnv, positions: dict[NodeId, tuple[int, int]]) -> None:
        pygame = self.pygame
        for truck_id, truck in env.trucks.items():
            x, y = self._truck_position(env, truck_id, positions)
            rect = pygame.Rect(x - 9, y - 7, 18, 14)
            pygame.draw.rect(self.screen, (37, 99, 235), rect, border_radius=3)
            pygame.draw.rect(self.screen, (17, 24, 39), rect, width=2, border_radius=3)
            label = self.small_font.render(str(truck_id), True, (255, 255, 255))
            self.screen.blit(label, (x - 4, y - 8))

    def _draw_panel(
        self,
        env: GraphEnv,
        metrics: EpisodeMetrics,
        protagonist_reward: float,
        antagonist_reward: float,
        antagonist_action: dict[EdgeId, float],
        feed: list[tuple[str, str]],
    ) -> None:
        pygame = self.pygame
        panel_x = self.width - self.panel_width
        pygame.draw.rect(self.screen, (238, 242, 247), pygame.Rect(panel_x, 0, self.panel_width, self.height))
        lines = [
            "SACRED Toy SDVRP",
            f"Tick: {metrics.ticks}",
            "Unit: 1 tick = 1 s",
            f"Status: {metrics.done_reason}",
            f"Speed: {self.sim_ticks_per_second:.1f} ticks/s",
            f"Mode: {'paused' if self.paused else 'running'}",
            f"Delivered: {metrics.total_delivery:.0f}",
            f"Distance: {metrics.total_distance:.1f}",
            f"P return: {metrics.protagonist_return:.2f}",
            f"A return: {metrics.antagonist_return:.2f}",
            f"P reward: {protagonist_reward:.2f}",
            f"A reward: {antagonist_reward:.2f}",
            f"Budget used: {metrics.congestion_budget_used:.1f}",
            f"Events: {metrics.congestion_events}",
        ]
        if antagonist_action:
            edge, level = next(iter(antagonist_action.items()))
            lines.append(f"Last attack: {edge} @ {level:.2f}")
        else:
            lines.append("Last attack: none")

        y = 28
        for index, line in enumerate(lines):
            font = self.font if index == 0 else self.small_font
            color = (17, 24, 39) if index == 0 else (51, 65, 85)
            self.screen.blit(font.render(line, True, color), (panel_x + 24, y))
            y += 34 if index == 0 else 24

        y += 8
        self.screen.blit(self.small_font.render("Fleet", True, (17, 24, 39)), (panel_x + 24, y))
        y += 22
        for truck_id, truck in env.trucks.items():
            location = truck.current_node if truck.current_node is not None else f"{truck.edge[0]}-{truck.edge[1]}"
            line = f"Truck {truck_id}: load {truck.load:.0f}/{truck.capacity:.0f}, at {location}"
            self.screen.blit(self.tiny_font.render(line, True, (51, 65, 85)), (panel_x + 24, y))
            y += 16

        y += 10
        self.screen.blit(self.small_font.render("Agent Feed", True, (17, 24, 39)), (panel_x + 24, y))
        y += 24
        for role, message in feed[-10:]:
            color = (37, 99, 235) if role == "P" else (190, 18, 60)
            label = self.tiny_font.render(role, True, color)
            self.screen.blit(label, (panel_x + 24, y))
            for line in self._wrap_text(message, 38):
                text = self.tiny_font.render(line, True, (51, 65, 85))
                self.screen.blit(text, (panel_x + 46, y))
                y += 16
            y += 2

        controls = [
            "Controls",
            "Space pause/resume",
            "Up/+ faster",
            "Down/- slower",
            "Right step while paused",
            "Esc quit",
        ]
        y = self.height - 142
        for index, line in enumerate(controls):
            font = self.small_font if index == 0 else self.tiny_font
            color = (17, 24, 39) if index == 0 else (71, 85, 105)
            self.screen.blit(font.render(line, True, color), (panel_x + 24, y))
            y += 20

    def _truck_position(
        self,
        env: GraphEnv,
        truck_id: int,
        positions: dict[NodeId, tuple[int, int]],
    ) -> tuple[int, int]:
        truck = env.trucks[truck_id]
        if truck.current_node is not None:
            return positions[truck.current_node]
        if truck.edge is None:
            return positions[env.depot_node]

        u, v = truck.edge
        start = positions[u]
        end = positions[v]
        edge_distance = env.graph.edges[u, v]["distance"]
        ratio = min(1.0, max(0.0, truck.edge_progress / edge_distance))
        x = int(start[0] + ((end[0] - start[0]) * ratio))
        y = int(start[1] + ((end[1] - start[1]) * ratio))
        return x, y

    def _congestion_color(self, congestion: float) -> Color:
        free = (148, 163, 184)
        blocked = (190, 18, 60)
        ratio = min(1.0, max(0.0, congestion))
        return tuple(int(free[i] + ((blocked[i] - free[i]) * ratio)) for i in range(3))

    def _wrap_text(self, text: str, width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
