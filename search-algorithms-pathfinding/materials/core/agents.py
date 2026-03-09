from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Callable

from core.grid import Grid
from core.path import Path


@dataclass(slots=True)
class Agent:
    name: str

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        raise NotImplementedError


class ExampleAgent(Agent):

    def __init__(self):
        super().__init__("Example")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        nodes = [start]
        while nodes[-1] != goal:
            r, c = nodes[-1]
            neighbors = grid.neighbors4(r, c)   #find all neigbors
            
            #for each surrounding tile, check which is the closest to goal
            min_dist = min(grid.manhattan(t.pos, goal) for t in neighbors)
            #if two or more are with the same distance from the goal, choose one randomly
            best_tiles = [
                tile for tile in neighbors
                if grid.manhattan(tile.pos, goal) == min_dist
            ]
            best_tile = best_tiles[random.randint(0, len(best_tiles) - 1)]

            nodes.append(best_tile.pos)

        return Path(nodes)


class DFSAgent(Agent):

    def __init__(self):
        super().__init__("DFS")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        """
        visited = set()
        visited.add(grid.get(start[0], start[1]))
        stack = [start]
        path = [start]

        while stack != goal:
            curr = stack.pop()
            if curr == goal:
                break
            r, c = curr
            options = []
            #for all neighbors search for the tile with best price
            for tile in grid.neighbors4DFS(r, c):
                if tile not in visited:
                    options.append(tile)

            if not options:
                break

            #search through each option and find the one with the lowest cost, almost_best_tile/s
            min_cost = min(tile.cost for tile in options)
            almost_best_tile = [tile for tile in options if tile.cost == min_cost]
            #if two or more with the same cost, pick the first
            #because it is already sorted by directions in neighbors4DFS, E,S,W,N
            best_tile = almost_best_tile[0]

            visited.add(best_tile)
            path.append(best_tile.pos)
            stack.append(best_tile.pos)

        return Path(path)
        """


        expanded = 0
        visited = set()
        stack = [(start, [start])]  # store tuples: (current_pos, path_so_far)

        while stack:
            current, path = stack.pop()
            expanded += 1
            if current == goal:
                print(expanded)
                return Path(path)  # goal reached

            if current in visited:
                continue
            visited.add(current)

            r, c = current
            # get unvisited neighbors, neighbors is a list of Tiles
            neighbors = [tile for tile in grid.neighbors4DFS(r, c) if tile.pos not in visited]

            if not neighbors:
                continue  # no neighbors, end of map or no unvisited tiles, try next tile in stack

            # sort neighbors by cost
            neighbors.sort(key=lambda t: t.cost)

            # push all neighbors to stack in reverse order so cheapest is popped first
            for tile in reversed(neighbors):
                stack.append((tile.pos, path + [tile.pos]))

        # if goal is not reached
        return Path([])


class BranchAndBoundAgent(Agent):

    def __init__(self):
        super().__init__("BranchAndBound")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        # Priority queue:
        # (total_cost_of_the_path, path_length_in_tiles, random_val, visited_tiles_on_path)
        priorQue = []
        heapq.heappush(priorQue, (0, 1, random.random(), [start]))
        expanded = 0

        best_cost = float("inf")
        #best_path = None
        while priorQue:
            # pop the one with lowest cost
            # if two or more have the same cost, pop the one with lowest length
            # else it does not matter which one is popped, the result will have the minimal cost
            cost, length, randNum, path = heapq.heappop(priorQue)
            current = path[-1]
            expanded += 1
            # Bounding: prune paths that are already worse
            # if cost >= best_cost:
            #    continue

            # If goal reached, update best solution
            if current == goal:
                # best_cost = cost
                # best_path = path
                # continue
                print("Expanded nodes: ", expanded)
                return Path(path)

            r, c = current
            for tile in grid.neighbors4(r, c):
                next_pos = tile.pos

                # Avoid cycles
                if next_pos in path:
                    continue

                new_cost = cost + tile.cost
                new_path = path + [next_pos]

                heapq.heappush(
                    priorQue,
                    (new_cost, len(new_path), random.random(), new_path)
                )

        #return Path(best_path)
        # If no path exists
        return Path([])


class AStar(Agent):

    def __init__(self):
        super().__init__("AStar")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        expanded = 0
        # (evaluation_func, path_length_in_tiles, random_val, total_cost_of_the_path, visited_tiles_on_path)
        priorQue = []
        heur_start = grid.manhattan(start, goal)    # manhattan distance from the goal as heuristic function
        heapq.heappush(priorQue, (heur_start, 1, random.random(), 0, [start]))
        # best_g = {start: 0}
        while priorQue:
            f, length, randomNum, g, path = heapq.heappop(priorQue)
            current = path[-1]
            expanded += 1

            # Goal reached → optimal path
            if current == goal:
                print("Expanded nodes: ", expanded)
                return Path(path)

            r, c = current
            for tile in grid.neighbors4(r, c):
                next_pos = tile.pos

                # Avoid cycles
                if next_pos in path:
                    continue

                new_g = g + tile.cost

                # if next_pos in best_g and new_g >= best_g[next_pos]:
                #     continue

                # best_g[next_pos] = new_g
                new_h = grid.manhattan(next_pos, goal)
                new_f = new_g + new_h   # eval_func = h + g
                new_path = path + [next_pos]

                heapq.heappush(
                    priorQue,
                    (new_f, len(new_path), random.random(), new_g, new_path)
                )

        # If no path exists
        return Path([])


AGENTS: dict[str, Callable[[], Agent]] = {
    "Example": ExampleAgent,
    "DFS": DFSAgent,
    "BranchAndBound": BranchAndBoundAgent,
    "AStar": AStar
}


def create_agent(name: str) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Available: {', '.join(AGENTS.keys())}")
    return AGENTS[name]()
