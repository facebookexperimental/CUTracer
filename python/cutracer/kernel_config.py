# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Kernel Configuration Abstraction for Trace Metadata

"""
Kernel-level configuration extracted from trace metadata.

Provides KernelConfig dataclass that captures kernel launch parameters
from the kernel_metadata event in CUTracer trace files.
"""

from dataclasses import dataclass

from cutracer.types import TraceRecord

THREADS_PER_WARP = 32
WARPS_PER_WARPGROUP = 4  # Hopper (SM90) warpgroup size


@dataclass(frozen=True)
class KernelConfig:
    """
    Kernel-level configuration extracted from trace metadata.

    Combines static metadata (from kernel_metadata event) with
    derived properties computed from launch parameters.

    Attributes:
        kernel_name: Unmangled kernel function name
        kernel_checksum: Binary fingerprint for kernel identification
        block_dims: Threads per CTA as (x, y, z) tuple
        grid_dims: CTAs per grid as (x, y, z) tuple
        shmem_dynamic_bytes: Dynamic shared memory allocation size
        shmem_static_bytes: Static shared memory allocation size
        nregs: Register usage per thread
        cubin_path: Relative path to dumped cubin file (only when dump_cubin enabled)
    """

    kernel_name: str
    kernel_checksum: str
    block_dims: tuple[int, int, int]
    grid_dims: tuple[int, int, int]
    shmem_dynamic_bytes: int
    shmem_static_bytes: int = 0
    nregs: int = 0
    cubin_path: str = ""  # Only set when dump_cubin is enabled
    sm_family: int = 0  # SM architecture family (e.g., 90=Hopper, 100=Blackwell)
    cluster_dims: tuple[int, int, int] = (1, 1, 1)
    cluster_dim_source: str = "unknown"

    @property
    def threads_per_cta(self) -> int:
        """Total threads per CTA (block_dims product)."""
        return self.block_dims[0] * self.block_dims[1] * self.block_dims[2]

    @property
    def warps_per_cta(self) -> int:
        """Number of warps per CTA."""
        return (self.threads_per_cta + THREADS_PER_WARP - 1) // THREADS_PER_WARP

    @property
    def warpgroups_per_cta(self) -> int:
        """Number of warpgroups per CTA."""
        return (self.warps_per_cta + WARPS_PER_WARPGROUP - 1) // WARPS_PER_WARPGROUP

    @property
    def total_shmem_bytes(self) -> int:
        """Total shared memory (dynamic + static)."""
        return self.shmem_dynamic_bytes + self.shmem_static_bytes

    @property
    def total_ctas(self) -> int:
        """Total CTAs in grid (grid_dims product)."""
        return self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]

    @property
    def cluster_size(self) -> int:
        """Number of CTAs in one launch cluster."""
        return self.cluster_dims[0] * self.cluster_dims[1] * self.cluster_dims[2]

    def global_warps_for_cta(self, cta: tuple[int, int, int]) -> range:
        """Return NVBit's grid-global warp-id range for one CTA.

        NVBit's ``get_global_warp_id`` linearizes CTAs x-fastest, then
        reserves ``ceil(block_threads / 32)`` consecutive ids per CTA.

        Raises:
            ValueError: If launch dimensions are unavailable or ``cta`` lies
                outside the configured grid.
        """
        if any(dimension <= 0 for dimension in (*self.block_dims, *self.grid_dims)):
            raise ValueError("block and grid dimensions must be positive")
        grid_x, grid_y, grid_z = self.grid_dims
        cta_x, cta_y, cta_z = cta
        if not (0 <= cta_x < grid_x and 0 <= cta_y < grid_y and 0 <= cta_z < grid_z):
            raise ValueError(f"CTA {cta} lies outside launch grid {self.grid_dims}")
        linear_cta = cta_x + cta_y * grid_x + cta_z * grid_x * grid_y
        first_warp = linear_cta * self.warps_per_cta
        return range(first_warp, first_warp + self.warps_per_cta)


def parse_kernel_metadata(record: TraceRecord) -> KernelConfig | None:
    """
    Parse kernel_metadata event into KernelConfig.

    The kernel_metadata event is the first event in new-format CUTracer
    trace files, containing kernel launch parameters captured via CUDA
    driver API.

    Args:
        record: First event from trace file

    Returns:
        KernelConfig if record is kernel_metadata type, None otherwise

    Example:
        >>> record = {"type": "kernel_metadata", "block": [384, 1, 1], ...}
        >>> config = parse_kernel_metadata(record)
        >>> config.warps_per_cta
        12
    """
    if record.get("type") != "kernel_metadata":
        return None

    block = record.get("block", [0, 0, 0])
    grid = record.get("grid", [0, 0, 0])
    cluster = record.get("cluster_dim", [1, 1, 1])

    return KernelConfig(
        kernel_name=record.get("unmangled_name", record.get("mangled_name", "")),
        kernel_checksum=record.get("kernel_checksum", ""),
        block_dims=(block[0], block[1], block[2]),
        grid_dims=(grid[0], grid[1], grid[2]),
        shmem_dynamic_bytes=record.get("shmem_dynamic", 0),
        shmem_static_bytes=record.get("shmem_static", 0),
        nregs=record.get("nregs", 0),
        cubin_path=record.get("cubin_path", ""),
        sm_family=record.get("sm_family", 0),
        cluster_dims=(cluster[0], cluster[1], cluster[2]),
        cluster_dim_source=record.get("cluster_dim_source", "unknown"),
    )
