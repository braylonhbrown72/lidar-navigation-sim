#!/usr/bin/env python3
"""LiDAR Grid Random-Walk Navigation Demo

Loads a synthetic room occupancy grid (NPZ) and simulates a LiDAR-equipped
robot doing a random walk with simple reactive obstacle avoidance.

The inflated occupancy grid (obstacles padded by the robot's radius plus a
safety buffer) is used as the navigation costmap. The LiDAR is simulated by
raycasting through the grid.

Outputs
-------
- PNG plot of the traversed path
- Animated GIF of the robot moving through the room

Usage
-----
    python lidar_nav_sim.py --data lidar_room_grid.npz --steps 400 --seed 123
"""

import argparse
import json
import math

import imageio
import matplotlib
matplotlib.use("Agg")  # render without opening a window
import matplotlib.pyplot as plt
import numpy as np


def load_map(path):
    d = np.load(path, allow_pickle=False)
    x_centers = d['x_centers']; y_centers = d['y_centers']
    floor_z = d['floor_z']; max_z = d['max_z']
    occ = d['occ']; inflated_occ = d['inflated_occ']
    meta = json.loads(str(d['metadata']))
    return x_centers, y_centers, floor_z, max_z, occ, inflated_occ, meta


def world_to_grid(x, y, res):
    ix = int(np.floor(x / res))
    iy = int(np.floor(y / res))
    return iy, ix


def grid_to_world(iy, ix, res):
    x = (ix + 0.5) * res
    y = (iy + 0.5) * res
    return x, y


def raycast(x, y, theta, nav_grid, meta, max_range=10.0):
    """March along one beam until it leaves the room or hits an occupied cell."""
    res = meta['resolution_m']; room_x, room_y = meta['room_size_m']
    step = res / 2.0
    dist = 0.0
    while dist < max_range:
        rx = x + dist * math.cos(theta)
        ry = y + dist * math.sin(theta)
        if not (0 <= rx < room_x and 0 <= ry < room_y):
            return dist
        iy = int(np.floor(ry / res)); ix = int(np.floor(rx / res))
        if nav_grid[iy, ix] != 0:
            return dist
        dist += step
    return max_range


def lidar_scan(x, y, yaw, nav_grid, meta, fov=2*math.pi, n_beams=180, max_range=10.0):
    """Simulate a full LiDAR sweep: one raycast per beam angle."""
    start_ang = yaw - fov/2
    angles = np.linspace(start_ang, start_ang + fov, n_beams, endpoint=False)
    ranges = np.array([raycast(x, y, ang, nav_grid, meta, max_range) for ang in angles])
    rel_angles = angles - yaw
    return rel_angles, ranges


def choose_heading(rel_angles, ranges, rng, avoid_dist=0.35):
    """If something is close in the front +/-30 deg arc, turn toward the most open
    direction. Otherwise, keep going with a small random jitter."""
    mask = np.abs(rel_angles) < np.deg2rad(30)
    if mask.any() and ranges[mask].min() < avoid_dist:
        idx = np.argmax(ranges)
        return rel_angles[idx]
    return np.deg2rad(rng.uniform(-10, 10))


def step_robot(x, y, yaw, v=0.05, turn=0.0):
    yaw_n = yaw + turn
    return x + v * math.cos(yaw_n), y + v * math.sin(yaw_n), yaw_n


def simulate(path, steps=400, seed=123, start=(0.3, 2.0, 0.0),
             avoid_dist=0.35, step_dist=0.05):
    x_c, y_c, floor_z, max_z, occ, inflated, meta = load_map(path)
    nav_grid = inflated.astype(np.uint8)
    res = meta['resolution_m']; room_x, room_y = meta['room_size_m']

    rng = np.random.default_rng(seed)
    x, y, yaw = start
    # If the start position is occupied, pick a random free cell instead
    iy, ix = world_to_grid(x, y, res)
    if not (0 <= ix < nav_grid.shape[1] and 0 <= iy < nav_grid.shape[0]) or nav_grid[iy, ix] != 0:
        free = np.argwhere(nav_grid == 0)
        iy, ix = free[rng.integers(len(free))]
        x, y = grid_to_world(iy, ix, res)

    poses = np.zeros((steps, 3))
    hits = np.zeros(steps, dtype=bool)
    for t in range(steps):
        rel, rngs = lidar_scan(x, y, yaw, nav_grid, meta, fov=2*math.pi, n_beams=180)
        turn = choose_heading(rel, rngs, rng, avoid_dist=avoid_dist)
        x_n, y_n, yaw_n = step_robot(x, y, yaw, v=step_dist, turn=turn)
        # Block any move that would leave the room or enter an obstacle
        if not (0 <= x_n < room_x and 0 <= y_n < room_y):
            hits[t] = True
            yaw_n = yaw + rng.uniform(-math.pi/2, math.pi/2)
            x_n, y_n = x, y
        else:
            iy, ix = world_to_grid(x_n, y_n, res)
            if nav_grid[iy, ix] != 0:
                hits[t] = True
                yaw_n = yaw + rng.uniform(-math.pi/2, math.pi/2)
                x_n, y_n = x, y
        x, y, yaw = x_n, y_n, yaw_n
        poses[t] = (x, y, yaw)
    return poses, hits, nav_grid, meta


def plot_path(poses, hits, nav_grid, meta, out_png):
    room_x, room_y = meta['room_size_m']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(nav_grid, origin='lower', extent=[0, room_x, 0, room_y],
              cmap='Reds', alpha=0.5)
    ax.plot(poses[:, 0], poses[:, 1], 'k-', lw=1)
    ax.plot(poses[0, 0], poses[0, 1], 'go', label='Start')
    ax.plot(poses[-1, 0], poses[-1, 1], 'bo', label='End')
    coll = poses[hits, :2]
    if coll.size:
        ax.plot(coll[:, 0], coll[:, 1], 'rx', label='Avoided Hit')
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Random-Walk LiDAR Nav Path')
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def animate(poses, nav_grid, meta, out_gif, stride=20):
    room_x, room_y = meta['room_size_m']
    frames = []
    for t in range(0, len(poses), stride):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(nav_grid, origin='lower', extent=[0, room_x, 0, room_y],
                  cmap='Reds', alpha=0.5)
        ax.plot(poses[:t+1, 0], poses[:t+1, 1], 'k-', lw=1)
        ax.plot(poses[t, 0], poses[t, 1], 'go', ms=4)
        ax.set_xlim(0, room_x); ax.set_ylim(0, room_y)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())
        plt.close(fig)
    imageio.mimsave(out_gif, frames, duration=0.15, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='lidar_room_grid.npz')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--png', default='lidar_nav_path.png')
    ap.add_argument('--gif', default='lidar_nav_anim.gif')
    ap.add_argument('--stride', type=int, default=20,
                    help='frame stride for lightweight GIF')
    args = ap.parse_args()

    poses, hits, nav_grid, meta = simulate(args.data, steps=args.steps, seed=args.seed)
    plot_path(poses, hits, nav_grid, meta, args.png)
    animate(poses, nav_grid, meta, args.gif, stride=args.stride)
    print('Saved:', args.png, args.gif)


if __name__ == '__main__':
    main()
