# Spatial Grid Algorithm

The Spatial Grid is a 2D spatial indexing structure used to accelerate collision detection and neighbor queries in the optimization algorithms. It partitions the warehouse floor into cells to speed up intersection checks.

## Key Features
- **Grid Partitioning**: Divides the warehouse floor into equal-sized cells (default 2x2 or 4x4 units).
- **Efficient Lookup**: Reduces the time complexity of item-item overlap queries from O(N^2) to near O(1).
- **Spatial Indexing**: Stores sets of item indices that intersect with each grid cell.

## Code Snippet (Python Implementation)

```python
import math

class SimpleGrid:
    """A 2D Spatial Grid to speed up item-item overlap and gravity checks."""
    def __init__(self, wh_l, wh_w, cell_size=2.0):
        self.cell_size = cell_size
        self.cols = max(1, math.ceil(wh_l / cell_size))
        self.rows = max(1, math.ceil(wh_w / cell_size))
        # Grid of sets containing indices of placed items
        self.grid = [[set() for _ in range(self.rows)] for _ in range(self.cols)]

    def _get_cells(self, x1, y1, x2, y2):
        c1 = max(0, min(self.cols-1, int(x1 / self.cell_size)))
        c2 = max(0, min(self.cols-1, int(x2 / self.cell_size)))
        r1 = max(0, min(self.rows-1, int(y1 / self.cell_size)))
        r2 = max(0, min(self.rows-1, int(y2 / self.cell_size)))
        return c1, c2, r1, r2

    def insert(self, idx, x1, y1, x2, y2):
        """Insert item index into occupied grid cells."""
        c1, c2, r1, r2 = self._get_cells(x1, y1, x2, y2)
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                self.grid[c][r].add(idx)

    def query(self, x1, y1, x2, y2):
        """Query for items that intersect with a given bounding box."""
        c1, c2, r1, r2 = self._get_cells(x1, y1, x2, y2)
        matches = set()
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                matches.update(self.grid[c][r])
        return matches
```
