"""
path_index.py  —  Pre-computes the index arrays that
AffinityDisplacementLoss needs for vectorised Eq (7) and Eq (5).

Usage
-----
    from path_index import PathIndex
    pi = PathIndex(feat_h=128, feat_w=128, radius=5)
    # then pass pi to AffinityDisplacementLoss(pi)

What it computes
----------------
search_dst  : (num_directions, 2)  ndarray  —  (dy, dx) for every
              neighbour direction within the search radius.
path_indices: list of ndarrays, one per distinct path length.
              Each array has shape (num_pairs, path_length, 1) and
              contains flat pixel indices along the line from src to dst.
              AffinityDisplacementLoss gathers the boundary map at these
              indices and max-pools over the path_length dim  →  Eq (7).
radius_floor: int  —  floor(radius).  Used by to_pair_displacement()
              to know how much to crop from the edges of the feature map.
"""

import numpy as np
from collections import defaultdict


class PathIndex:

    def __init__(self, feat_h, feat_w, radius=5):
        """
        Parameters
        ----------
        feat_h, feat_w : int
            Spatial size of the IRNet output feature maps (typically input/4).
        radius : int
            Search radius for neighbour pairs.  All integer (dy, dx) with
            0 < sqrt(dy² + dx²) ≤ radius are included.
        """
        self.feat_h      = feat_h
        self.feat_w      = feat_w
        self.radius      = radius
        self.radius_floor = int(np.floor(radius))

        # ── enumerate all neighbour offsets within radius ──────────
        offsets = []
        for dy in range(-self.radius_floor, self.radius_floor + 1):
            for dx in range(-self.radius_floor, self.radius_floor + 1):
                if dy == 0 and dx == 0:
                    continue
                if np.sqrt(dy * dy + dx * dx) <= radius:
                    # only keep offsets where dy > 0, or (dy==0 and dx > 0)
                    # to avoid duplicate pairs (i,j) and (j,i)
                    if dy > 0 or (dy == 0 and dx > 0):
                        offsets.append((dy, dx))

        self.search_dst = np.array(offsets, dtype=np.int64)       # (num_dir, 2)

        # ── build path indices ─────────────────────────────────────
        self._build_path_indices()

    def _build_path_indices(self):
        """
        For every (dy, dx) in search_dst and every valid source pixel,
        walk the Bresenham line from src to dst and record the flat indices
        of all pixels on that line.  Group by path length so that each
        group can be max-pooled with a single F.max_pool2d call.
        """
        h, w = self.feat_h, self.feat_w

        # group paths by their length
        paths_by_length = defaultdict(list)   # length → list of index-arrays

        for dy, dx in self.search_dst:
            # valid source region (so that dst stays in-bounds)
            y_start = max(0, -dy)
            y_end   = min(h, h - dy)
            x_start = max(0, -dx)
            x_end   = min(w, w - dx)

            for sy in range(y_start, y_end):
                for sx in range(x_start, x_end):
                    # Bresenham line from (sy, sx) to (sy+dy, sx+dx)
                    line = self._bresenham(sy, sx, sy + dy, sx + dx)
                    flat_indices = [r * w + c for r, c in line]
                    paths_by_length[len(flat_indices)].append(flat_indices)

        # convert to numpy arrays grouped by length
        # shape per group: (num_pairs, path_len, 1)
        self.path_indices = []
        for length in sorted(paths_by_length.keys()):
            arr = np.array(paths_by_length[length], dtype=np.int64)  # (N, L)
            arr = arr[:, :, None]                                     # (N, L, 1)
            self.path_indices.append(arr)

    @staticmethod
    def _bresenham(y0, x0, y1, x1):
        """Integer Bresenham line from (y0,x0) to (y1,x1), inclusive."""
        points = []
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y1 > y0 else -1
        sx = 1 if x1 > x0 else -1
        err = dx - dy

        cy, cx = y0, x0
        while True:
            points.append((cy, cx))
            if cy == y1 and cx == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx  += sx
            if e2 < dx:
                err += dx
                cy  += sy
        return points
