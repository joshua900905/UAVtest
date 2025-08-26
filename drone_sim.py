import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import linear_sum_assignment
import time

# (Constants and Path Planning are correct and remain unchanged)
ON_GROUND = 0
IN_AIR = 1
def generate_snake_paths(num_drones, area_width, area_height, grid_step=50, speed=2.5, ts=1.0):
    paths_waypoints = [[] for _ in range(num_drones)]
    strip_width = area_width / num_drones
    for i in range(num_drones):
        waypoints = [(0, 0)]
        x_start = i * strip_width
        y_points = np.arange(grid_step / 2, area_height, grid_step)
        for j, y in enumerate(y_points):
            if j % 2 == 0: waypoints.extend([(x_start + strip_width * 0.25, y), (x_start + strip_width * 0.75, y)])
            else: waypoints.extend([(x_start + strip_width * 0.75, y), (x_start + strip_width * 0.25, y)])
        waypoints.append((0, 0))
        paths_waypoints[i] = np.array(waypoints)
    interpolated_paths, max_len = [[] for _ in range(num_drones)], 0
    for i in range(num_drones):
        path, full_path = paths_waypoints[i], []
        for j in range(len(path) - 1):
            start_node, end_node = path[j], path[j+1]
            dist, num_steps = np.linalg.norm(end_node - start_node), int(np.ceil(np.linalg.norm(end_node - start_node) / (speed * ts)))
            if num_steps > 0: full_path.extend(zip(np.linspace(start_node[0], end_node[0], num_steps), np.linspace(start_node[1], end_node[1], num_steps)))
        interpolated_paths[i] = np.array(full_path) if full_path else np.array([path[-1]])
        if len(full_path) > max_len: max_len = len(full_path)
    for i in range(num_drones):
        path_len = len(interpolated_paths[i])
        if path_len < max_len: interpolated_paths[i] = np.vstack([interpolated_paths[i], np.tile(interpolated_paths[i][-1], (max_len - path_len, 1))])
    return interpolated_paths


class RelayPlanner:
    def __init__(self, rc, v_relay, ts, p_bs):
        self.rc, self.v_relay, self.ts, self.p_bs = rc, v_relay, ts, p_bs
        self.max_travel_dist = v_relay * ts

    def plan_tasks(self, p_s_next, p_r_all, r_states_all):
        """
        Main planning function, a direct and robust translation of the Fig. 1 workflow.
        """
        p_r_in_air = p_r_all[r_states_all == IN_AIR]
        p_init = None

        # Block I: ConnCheckPre
        if self._is_connected(p_s_next, p_r_in_air):
            p_init = p_r_in_air
        else:
            # Block II: MinRelay & MinCostTask -> P_temp
            p_candidates = self._min_relay_candidates(p_s_next, p_r_in_air)
            
            # The init assignment encourages movement towards strategically important (but far) points.
            assignments_init = self._get_assignments(p_r_all, r_states_all, p_candidates, mode='init')
            p_temp = self._predict_next_positions(p_r_all, r_states_all, p_candidates, assignments_init)
            
            # For the check, we need to know which of the p_temp drones are actually aerial.
            p_temp_in_air = self._get_in_air_positions(p_temp, r_states_all, assignments_init)

            # Block III & IV: ConnCheckPost
            if self._is_connected(p_s_next, p_temp_in_air):
                p_init = p_temp_in_air
            else:
                p_init = self._unite_arrays(p_temp_in_air, p_r_in_air)

        # Block V: Optimization
        p_opt = self._used_relay_set(p_s_next, p_init)
        
        # This is our robust fallback. If optimization fails, but connection is needed,
        # we must use the un-optimized (but safer) p_init.
        if (p_opt is None or p_opt.shape[0] == 0) and not self._is_connected(p_s_next, np.array([])):
            p_opt = p_init
        
        if p_opt is None or p_opt.shape[0] == 0:
            return {}

        # The opt assignment has hard constraints.
        final_assignments = self._get_assignments(p_r_all, r_states_all, p_opt, mode='opt')
        
        tasks = {r_idx: p_opt[t_idx] for r_idx, t_idx in final_assignments.items()}
        return tasks

    def _get_assignments(self, p_r_all, r_states_all, p_candidates, mode):
        num_relays, num_tasks = len(p_r_all), len(p_candidates)
        if num_tasks == 0: return {}
        cost_matrix = np.full((num_relays, num_tasks), np.inf)
        for r_idx in range(num_relays):
            start_pos = self.p_bs[0] if r_states_all[r_idx] == ON_GROUND else p_r_all[r_idx]
            dists = cdist(start_pos.reshape(1, -1), p_candidates)[0]
            
            # This logic now reflects Algorithm 3 (IdleDuration is simplified to 1 step for busy)
            if mode == 'init':
                # Soft penalty for unreachable targets, encouraging movement.
                cost_matrix[r_idx, :] = np.where(dists <= self.max_travel_dist, dists, dists * 2)
            elif mode == 'opt':
                # Hard constraint for unreachable targets.
                reachable = dists <= self.max_travel_dist
                cost_matrix[r_idx, reachable] = dists[reachable]
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return {r: c for r, c in zip(row_ind, col_ind) if np.isfinite(cost_matrix[r, c])}
        except ValueError:
             return {}
    
    def _predict_next_positions(self, p_r_all, r_states_all, p_targets, assignments):
        p_next = np.copy(p_r_all)
        if p_targets is None or p_targets.shape[0] == 0: return p_next
        for r_idx, target_idx in assignments.items():
            if target_idx >= len(p_targets): continue
            target_pos = p_targets[target_idx]
            start_pos = self.p_bs[0] if r_states_all[r_idx] == ON_GROUND else p_r_all[r_idx]
            direction = target_pos - start_pos
            distance = np.linalg.norm(direction)
            if distance > 1e-6: p_next[r_idx] = start_pos + (direction / distance) * min(distance, self.max_travel_dist)
        return p_next
    
    def _get_in_air_positions(self, p_all, r_states, assignments):
        # A relay is considered "in air" for the next step if it's currently in air, OR if it's on the ground but just got a task.
        newly_assigned_ground_relays = [r_idx for r_idx in assignments.keys() if r_states[r_idx] == ON_GROUND]
        in_air_mask = (r_states == IN_AIR) | np.isin(np.arange(len(p_all)), newly_assigned_ground_relays)
        return p_all[in_air_mask].reshape(-1, 2)

    def _unite_arrays(self, arr1, arr2):
        arr1, arr2 = np.array(arr1).reshape(-1, 2), np.array(arr2).reshape(-1, 2)
        if arr1.shape[0] == 0: return arr2
        if arr2.shape[0] == 0: return arr1
        return np.unique(np.vstack([arr1, arr2]).round(decimals=5), axis=0)

    def _is_connected(self, p_s, p_r):
        p_s, p_r = np.array(p_s).reshape(-1, 2), np.array(p_r).reshape(-1, 2)
        nodes, idx_map = self._build_graph_nodes([self.p_bs, p_s, p_r])
        if 1 not in idx_map: return True
        graph = csr_matrix(cdist(nodes, nodes) <= self.rc)
        dist_matrix = shortest_path(csgraph=graph, directed=False, indices=list(idx_map[0]))
        return np.all(dist_matrix[0, list(idx_map[1])] < np.inf)

    def _build_graph_nodes(self, node_sets):
        # (This robust version is correct and remains unchanged)
        all_nodes_list, idx_map, current_offset = [], {}, 0
        for i, s_raw in enumerate(node_sets):
            s = np.array(s_raw).reshape(-1, 2)
            if s.shape[0] > 0:
                all_nodes_list.append(s)
                idx_map[i] = range(current_offset, current_offset + s.shape[0])
                current_offset += s.shape[0]
        if not all_nodes_list: return np.empty((0, 2)), {}
        return np.vstack(all_nodes_list), idx_map

    def _used_relay_set(self, p_s_next, p_init):
        # (This robust version is correct and remains unchanged)
        if p_init is None or p_init.shape[0] == 0: return np.array([])
        p_init = np.array(p_init).reshape(-1, 2)
        nodes, idx_map = self._build_graph_nodes([self.p_bs, p_s_next, p_init])
        if not all(k in idx_map for k in [0, 1, 2]): return p_init
        bs_indices, s_indices, init_indices = idx_map[0], idx_map[1], idx_map[2]
        graph = csr_matrix(cdist(nodes, nodes) <= self.rc)
        dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=list(s_indices), return_predecessors=True)
        used_init_indices_in_nodes = set()
        for i, s_idx in enumerate(s_indices):
            path_dists_to_bs = dist_matrix[i, list(bs_indices)]
            if not np.any(np.isfinite(path_dists_to_bs)): continue
            closest_bs_idx = list(bs_indices)[np.argmin(path_dists_to_bs)]
            curr = closest_bs_idx
            for _ in range(len(nodes) + 1):
                used_init_indices_in_nodes.add(curr)
                if curr == s_idx: break
                next_node = predecessors[i, curr]
                if next_node == -9999: break
                curr = next_node
        final_relay_indices_in_nodes = used_init_indices_in_nodes.intersection(set(init_indices))
        if not final_relay_indices_in_nodes: return np.array([])
        return nodes[list(final_relay_indices_in_nodes)]

    def _min_relay_candidates(self, p_s_next, p_r_in_air):
        nodes_list = [self.p_bs, p_s_next, p_r_in_air]
        valid_nodes = [np.array(n).reshape(-1, 2) for n in nodes_list if n is not None and np.array(n).size > 0]
        if sum(len(n) for n in valid_nodes) < 3: return np.array([])
        try:
            return Voronoi(np.vstack(valid_nodes)).vertices
        except Exception:
            return np.array([])

def run_simulation():
    NUM_MISSION_DRONES, TOTAL_RELAY_DRONES = 2, 15
    AREA_WIDTH, AREA_HEIGHT = 400, 400
    RC, V_RELAY, V_MISSION, TS = 80, 10.0, 2.5, 1.0
    SIMULATION_STEPS = 500
    P_BS = np.array([[0, 0]])
    mission_paths = generate_snake_paths(NUM_MISSION_DRONES, AREA_WIDTH, AREA_HEIGHT, speed=V_MISSION, ts=TS)
    p_r_all = np.tile(P_BS, (TOTAL_RELAY_DRONES, 1))
    r_states_all = np.full(TOTAL_RELAY_DRONES, ON_GROUND)
    planner = RelayPlanner(rc=RC, v_relay=V_RELAY, ts=TS, p_bs=P_BS)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))

    for t in range(min(SIMULATION_STEPS, len(mission_paths[0]) - 2)):
        p_s_next = np.array([path[t+1] for path in mission_paths])
        tasks = planner.plan_tasks(p_s_next, p_r_all, r_states_all)
        
        for i in range(TOTAL_RELAY_DRONES):
            is_busy = i in tasks
            
            # This is the simplified "hover" logic.
            # We revert to this for maximum stability.
            if is_busy:
                target = tasks[i]
            else:
                target = p_r_all[i] # Hover at current position if idle
            
            # Physical Movement
            current_pos = p_r_all[i]
            if not np.array_equal(target, current_pos):
                direction = target - current_pos
                distance = np.linalg.norm(direction)
                move_dist = min(distance, V_RELAY * TS)
                p_r_all[i] = current_pos + (direction / distance) * move_dist

            # State Update: Only takeoff, no landing
            if is_busy and r_states_all[i] == ON_GROUND:
                r_states_all[i] = IN_AIR

        # Visualization
        p_s_current = np.array([path[t] for path in mission_paths])
        ax.cla()
        p_r_in_air = p_r_all[r_states_all == IN_AIR]
        p_r_on_ground = p_r_all[r_states_all == ON_GROUND]
        
        if p_s_current.shape[0] > 0:
            air_nodes = np.vstack([P_BS, p_s_current, p_r_in_air]) if p_r_in_air.shape[0] > 0 else np.vstack([P_BS, p_s_current])
            dists = cdist(air_nodes, air_nodes)
            lines = [[air_nodes[i], air_nodes[j]] for i in range(len(air_nodes)) for j in range(i+1, len(air_nodes)) if dists[i,j] <= RC]
            ax.add_collection(LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.8))
            ax.plot(P_BS[:, 0], P_BS[:, 1], 'ks', markersize=12, label=f'Base Station ({p_r_on_ground.shape[0]} relays)')
            ax.plot(p_s_current[:, 0], p_s_current[:, 1], 'bo', markersize=8, label='Mission UAVs')
            if p_r_in_air.shape[0] > 0: ax.plot(p_r_in_air[:, 0], p_r_in_air[:, 1], 'go', markersize=6, alpha=0.7, label=f'Relay UAVs in Air ({p_r_in_air.shape[0]})')
        
        ax.set_xlim(-20, AREA_WIDTH + 20); ax.set_ylim(-20, AREA_HEIGHT + 20)
        ax.set_title(f'Step: {t+1}/{SIMULATION_STEPS}'); ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right'); plt.draw(); plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_simulation()