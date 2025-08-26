import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import linear_sum_assignment
import time

# --- 階段零: 任務路徑規劃 (for Mission UAVs) ---
def generate_snake_paths(num_drones, area_width, area_height, grid_step=50):
    """
    生成蛇形路徑來模擬 mTSP 覆蓋整個區域。
    """
    paths = [[] for _ in range(num_drones)]
    strip_width = area_width / num_drones
    for i in range(num_drones):
        path = [(0, 0)]
        x_start = i * strip_width
        y_points = np.arange(grid_step / 2, area_height, grid_step)
        for j, y in enumerate(y_points):
            if j % 2 == 0:
                path.append((x_start + strip_width * 0.25, y))
                path.append((x_start + strip_width * 0.75, y))
            else:
                path.append((x_start + strip_width * 0.75, y))
                path.append((x_start + strip_width * 0.25, y))
        path.append((0, 0))
        paths[i] = np.array(path)
    return paths

# --- 核心演算法: 中繼定位 (for Relay UAVs, 嚴格遵循論文) ---
class RelayPositioningAlgorithm:
    def __init__(self, rc, v_relay, ts):
        self.rc, self.v_relay, self.ts = rc, v_relay, ts
        self.max_travel_dist = v_relay * ts

    def run_one_step(self, p_s_next, p_r_current, p_bs):
        if p_r_current is None or p_r_current.shape[0] == 0:
            return np.empty((0, 2))
        is_connected = self._conn_check_pre(p_s_next, p_r_current, p_bs)
        p_init = None
        if is_connected:
            p_init = p_r_current
        else:
            p_candidates = self._min_relay_busy_voronoi(p_s_next, p_r_current, p_bs)
            # 論文框架下的備用方案：如果 Voronoi 失敗，候選點就只有任務機本身
            if p_candidates is None or p_candidates.shape[0] == 0:
                p_candidates = p_s_next
            
            p_temp, _ = self._min_cost_task(p_r_current, p_candidates, mode='init')
            p_init = self._conn_check_post(p_s_next, p_r_current, p_temp, p_bs)
            
        p_r_next_targets = p_r_current
        if p_init is not None and p_init.shape[0] > 0:
            p_opt = self._used_relay_set(p_s_next, p_init, p_bs)
            if p_opt is not None and p_opt.shape[0] > 0:
                p_r_next_targets, _ = self._min_cost_task(p_r_current, p_opt, mode='opt')
        return p_r_next_targets

    def _build_graph_nodes(self, node_sets):
        valid_node_sets = [s for s in node_sets if s is not None and s.shape[0] > 0]
        if not valid_node_sets: return np.empty((0, 2)), {}
        all_nodes = np.vstack(valid_node_sets)
        indices = {i: range(start, start + s.shape[0]) for i, (start, s) in enumerate(zip(np.cumsum([0] + [len(s) for s in valid_node_sets[:-1]]), valid_node_sets))}
        return all_nodes, indices

    def _get_csgraph(self, all_nodes):
        return csr_matrix(cdist(all_nodes, all_nodes) <= self.rc)

    def _conn_check_pre(self, p_s_next, p_r_current, p_bs):
        if p_r_current.shape[0] == 0: return False
        nodes, idx = self._build_graph_nodes([p_bs, p_s_next, p_r_current])
        if 0 not in idx or 1 not in idx: return False
        dist_matrix = shortest_path(csgraph=self._get_csgraph(nodes), directed=False, indices=list(idx[1]))
        return np.all(dist_matrix[:, list(idx[0])] < np.inf)

    def _min_relay_busy_voronoi(self, p_s_next, p_r_current, p_bs):
        p_mst = np.vstack([p_bs, p_s_next, p_r_current])
        if p_mst.shape[0] < 4: return np.array([])
        try:
            vor = Voronoi(p_mst)
            return np.array([v for v in vor.vertices if not np.any(np.isinf(v))])
        except Exception:
            return np.array([])

    def _min_cost_task(self, p_r_current, p_candidates, mode):
        if p_r_current.shape[0] == 0 or p_candidates.shape[0] == 0:
            return p_r_current, None
        dist_matrix = cdist(p_r_current, p_candidates)
        cost_matrix = dist_matrix.copy()
        if mode == 'init':
            cost_matrix[dist_matrix > self.max_travel_dist] *= 2
        elif mode == 'opt':
            cost_matrix[dist_matrix > self.max_travel_dist] += 1e9
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p_r_next_targets = p_r_current.copy()
        assignments = {r: c for r, c in zip(row_ind, col_ind)}
        for r_idx, c_idx in assignments.items():
            direction = p_candidates[c_idx] - p_r_current[r_idx]
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                move_dist = min(distance, self.max_travel_dist)
                p_r_next_targets[r_idx] += (direction / distance) * move_dist
        return p_r_next_targets, assignments

    def _conn_check_post(self, p_s_next, p_r_current, p_temp, p_bs):
        nodes, idx = self._build_graph_nodes([p_bs, p_s_next, p_temp])
        if 0 not in idx or 1 not in idx: return np.vstack([p_temp, p_r_current])
        dist_matrix = shortest_path(csgraph=self._get_csgraph(nodes), directed=False, indices=list(idx[1]))
        return p_temp if np.all(dist_matrix[:, list(idx[0])] < np.inf) else np.vstack([p_temp, p_r_current, p_s_next])

    def _used_relay_set(self, p_s_next, p_init, p_bs):
        nodes, idx = self._build_graph_nodes([p_bs, p_s_next, p_init])
        if not all(k in idx for k in [0, 1, 2]): return np.array([])
        bs_idx, s_idx, init_idx = idx[0], idx[1], idx[2]
        _, predecessors = shortest_path(csgraph=self._get_csgraph(nodes), directed=False, indices=list(s_idx), return_predecessors=True)
        used_init_indices = set()
        for i in range(len(list(s_idx))):
            s_node_idx, target_node_idx = list(s_idx)[i], list(bs_idx)[0]
            curr = target_node_idx
            if predecessors[i, curr] == -9999 and curr != s_node_idx: continue
            path = {curr}
            while predecessors[i, curr] != -9999 and curr != s_node_idx:
                curr = predecessors[i, curr]; path.add(curr)
            if curr == s_node_idx:
                used_init_indices.update({list(init_idx).index(n) for n in path if n in init_idx})
        return p_init[list(used_init_indices)] if used_init_indices else np.array([])


# --- 模擬器 (最終版，採用正確的部署評估邏輯) ---
class Simulator:
    def __init__(self, params):
        self.params = params
        self.p_bs = np.array([[0, 0]])
        self.mission_paths = generate_snake_paths(params['num_mission_drones'], params['area_width'], params['area_height'])
        self.p_s_current = np.array([path[0] for path in self.mission_paths])
        self.s_target_waypoint_indices = [1] * params['num_mission_drones']
        self.p_r_current = np.empty((0, 2))
        self.total_relays_deployed = 0
        self.algorithm = RelayPositioningAlgorithm(params['comm_range'], params['v_relay'], params['ts'])
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.current_step = 0

    def _get_s_next_pos_and_update_state(self):
        p_s_next = []
        for i in range(self.params['num_mission_drones']):
            path, current_pos, target_idx = self.mission_paths[i], self.p_s_current[i], self.s_target_waypoint_indices[i]
            if target_idx >= len(path):
                p_s_next.append(current_pos); continue
            target_waypoint = path[target_idx]
            distance_to_target = np.linalg.norm(target_waypoint - current_pos)
            arrival_radius = self.params['v_mission'] * self.params['ts']
            if distance_to_target < arrival_radius:
                self.s_target_waypoint_indices[i] += 1
                new_target_idx = self.s_target_waypoint_indices[i]
                if new_target_idx >= len(path):
                    p_s_next.append(current_pos); continue
                target_waypoint = path[new_target_idx]
            direction = target_waypoint - current_pos
            dist = np.linalg.norm(direction)
            move_dist = min(dist, self.params['v_mission'] * self.params['ts'])
            next_pos = current_pos + (direction / dist) * move_dist if dist > 1e-6 else current_pos
            p_s_next.append(next_pos)
        return np.array(p_s_next)

    def run(self):
        plt.ion()
        for t in range(self.params['sim_steps']):
            self.current_step = t
            print(f"--- Timestep {t} ---")
            
            # 1. 更新搜索無人機狀態
            self.p_s_current = self._get_s_next_pos_and_update_state()

            # === 2. 規劃中繼無人機 (部署與移動) ===

            # 2a. 評估理想需求 (全新的、正確的邏輯)
            #    第一步：創造候選點 (論文的 MinRelay 步驟)
            p_candidates_ideal = self.algorithm._min_relay_busy_voronoi(self.p_s_current, self.p_r_current, self.p_bs)
            
            #    第二步：篩選最佳點 (論文的 UsedRelaySet 步驟)
            p_opt_ideal = self.algorithm._used_relay_set(self.p_s_current, p_candidates_ideal, self.p_bs)
            num_relays_needed = p_opt_ideal.shape[0] if p_opt_ideal is not None else 0

            # 2b. 檢查資源缺口
            num_relays_in_air = self.p_r_current.shape[0]
            
            # 2c. 部署新中繼 (如果理想需求 > 現有資源)
            if num_relays_needed > num_relays_in_air:
                num_to_deploy = num_relays_needed - num_relays_in_air
                num_to_deploy = min(num_to_deploy, self.params['max_relays'] - self.total_relays_deployed)
                if num_to_deploy > 0:
                    print(f"  [DEPLOYMENT] Ideal need ({num_relays_needed}) > In air ({num_relays_in_air}). Deploying {num_to_deploy}.")
                    new_relays = np.tile(self.p_bs, (num_to_deploy, 1))
                    self.p_r_current = np.vstack([self.p_r_current, new_relays])
                    self.total_relays_deployed += num_to_deploy
            
            # 2d. 計算所有 (包括新部署的) 中繼無人機的最終移動
            p_r_next = self.algorithm.run_one_step(self.p_s_current, self.p_r_current, self.p_bs)
            
            # === 3. 更新中繼無人機的最終位置狀態 ===
            self.p_r_current = p_r_next
            
            # === 4. 視覺化 ===
            self.visualize()
            time.sleep(0.01)
        
        print("\nSimulation Finished!")
        plt.ioff()
        plt.show()

    def visualize(self):
        self.ax.clear()
        for path in self.mission_paths:
            self.ax.plot(path[:, 0], path[:, 1], ':', color='gray', alpha=0.5)
        self.ax.plot(0, 0, 'ks', markersize=15, label='Base Station')
        if self.p_s_current.shape[0] > 0:
            self.ax.plot(self.p_s_current[:, 0], self.p_s_current[:, 1], 'bo', markersize=10, label='Mission UAVs')
        if self.p_r_current is not None and self.p_r_current.shape[0] > 0:
            self.ax.plot(self.p_r_current[:, 0], self.p_r_current[:, 1], 'ro', markersize=8, alpha=0.8, label='Relay UAVs')
        
        all_nodes = np.vstack([s for s in [self.p_bs, self.p_s_current, self.p_r_current] if s is not None and s.shape[0] > 0])
        if all_nodes.shape[0] > 1:
            dist_matrix = cdist(all_nodes, all_nodes)
            for i in range(len(all_nodes)):
                for j in range(i + 1, len(all_nodes)):
                    if dist_matrix[i, j] <= self.params['comm_range']:
                        self.ax.plot([all_nodes[i, 0], all_nodes[j, 0]], [all_nodes[i, 1], all_nodes[j, 1]], '-', color='green', alpha=0.4)
        
        self.ax.set_xlim(-50, self.params['area_width'] + 50); self.ax.set_ylim(-50, self.params['area_height'] + 50)
        self.ax.set_aspect('equal', adjustable='box'); self.ax.legend(loc='upper right')
        self.ax.set_title(f"Timestep: {self.current_step}, Relays in Air: {self.p_r_current.shape[0] if self.p_r_current is not None else 0}")
        plt.draw(); plt.pause(0.001)

if __name__ == '__main__':
    sim_params = {
        'num_mission_drones': 2, 'max_relays': 15,
        'area_width': 400, 'area_height': 400,
        'comm_range': 75, 'v_mission': 10, 'v_relay': 40,
        'ts': 1, 'sim_steps': 200,
    }

    simulator = Simulator(sim_params)
    simulator.run()