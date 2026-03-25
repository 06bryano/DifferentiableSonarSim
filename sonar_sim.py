# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:39:55 2026

@author: ob300
"""

"""
Differentiable Side-Scan Sonar Renderer
========================================
Renders a synthetic sonar image from a depth map using physically motivated
components: Lambertian backscatter, triangle-mesh area weighting, range-based
intensity spreading, and differentiable occlusion (acoustic shadowing).

All operations are implemented in PyTorch so that gradients flow through the
renderer — enabling its use inside an optimisation / learning loop.

Coordinate convention
---------------------
- i-axis  : cross-track (x in the mesh)  — increases left→right in the depth map
- j-axis  : along-track (y in the mesh)  — sonar is at the far end (high j)
- z-axis  : vertical height of the seabed

Mesh tessellation (per depth-map quad)
---------------------------------------
Each 2×2 block of depth-map pixels is split into two triangles:

    B ---- C      (j = low)
    | ╲ 1  |
    |  ╲   |
    | 2 ╲  |
    A ---- D      (j = high)

    Triangle 1 (ABC): vertices A, B, C
    Triangle 2 (ACD): vertices A, C, D
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: mesh geometry
# ---------------------------------------------------------------------------

def _build_mesh_triangles(depth_map: torch.Tensor):
    """
    Tessellate the depth map into a grid of ABC / ACD triangles.

    For every quad (i, j) → (i+1, j+1) the four corner heights are:
        A = depth_map[j+1, i]   (bottom-left)
        B = depth_map[j,   i]   (top-left)
        C = depth_map[j,   i+1] (top-right)
        D = depth_map[j+1, i+1] (bottom-right)

    Returns
    -------
    mesh_heights : (2*H-2, W-1)  interleaved ABC / ACD centroid heights
    normals      : (3, 2*H-2, W-1)  unit surface normals
    areas        : (2*H-2, W-1)  triangle areas
    """
    H, W = depth_map.shape

    # --- vertex grids (shared by both triangle families) ---
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing='ij'
    )

    # 3-D vertex positions (x, y, z) for each corner
    A = torch.stack((xx[1:, :-1], yy[1:, :-1], depth_map[1:, :-1]))   # bottom-left
    B = torch.stack((xx[:-1, :-1], yy[:-1, :-1], depth_map[:-1, :-1]))  # top-left
    C = torch.stack((xx[:-1, 1:], yy[:-1, 1:], depth_map[:-1, 1:]))    # top-right
    D = torch.stack((xx[1:, 1:], yy[1:, 1:], depth_map[1:, 1:]))       # bottom-right

    # --- centroid heights ---
    heights_abc = (depth_map[1:, :-1] + depth_map[:-1, :-1] + depth_map[:-1, 1:]) / 3
    heights_acd = (depth_map[1:, :-1] + depth_map[:-1, 1:] + depth_map[1:, 1:]) / 3

    # --- surface normals via cross product ---
    AB = torch.subtract(A, B)
    AC = torch.subtract(A, C)
    CA = torch.subtract(C, A)
    CD = torch.subtract(C, D)

    cross_abc = torch.cross(AB, AC)                             # Triangle ABC
    norm_abc  = torch.norm(cross_abc, dim=0, keepdim=True)
    normals_abc = torch.divide(cross_abc, norm_abc)
    areas_abc   = 0.5 * norm_abc[0]

    cross_acd = torch.cross(CD, CA)                             # Triangle ACD
    norm_acd  = torch.norm(cross_acd, dim=0, keepdim=True)[0]
    normals_acd = torch.divide(cross_acd, norm_acd)
    areas_acd   = 0.5 * norm_acd

    # --- interleave ABC (even rows) and ACD (odd rows) ---
    n_rows = heights_abc.shape[0] + heights_acd.shape[0]
    n_cols = heights_abc.shape[1]
    dtype  = depth_map.dtype

    mesh_heights = torch.empty((n_rows, n_cols), dtype=dtype)
    mesh_heights[0::2, :] = heights_abc
    mesh_heights[1::2, :] = heights_acd

    normals = torch.empty((3, n_rows, n_cols), dtype=dtype)
    normals[:, 0::2, :] = normals_abc
    normals[:, 1::2, :] = normals_acd

    areas = torch.empty((n_rows, n_cols), dtype=dtype)
    areas[0::2, :] = areas_abc
    areas[1::2, :] = areas_acd

    return mesh_heights, normals, areas


# ---------------------------------------------------------------------------
# Helper: mesh spatial positions & sonar unit vectors
# ---------------------------------------------------------------------------

def _build_sonar_unit_vectors(
    depth_map: torch.Tensor,
    mesh_heights: torch.Tensor,
    sonar_y: float,
    sonar_z: float,
):
    """
    Compute the unit vector from each triangle centroid toward the sonar,
    plus a leading-edge variant used for the occlusion calculation.

    The sonar is modelled as a point source with no x-offset (2-D geometry
    in the y-z plane), so dx = 0 throughout.

    Parameters
    ----------
    depth_map     : original (H, W) depth map
    mesh_heights  : (2*H-2, W-1) interleaved centroid heights
    sonar_y, sonar_z : sonar position in pixel units

    Returns
    -------
    sonar_unit_vec  : (3, n_rows, n_cols)  unit vectors (x, y, z) to sonar
    suv_z_occl      : (n_rows, n_cols)     z-component using leading-edge heights
    slant_range     : (n_rows, n_cols)     distance from centroid to sonar
    """
    n_rows, n_cols = mesh_heights.shape

    # --- y (along-track) centroid positions ---
    # Each pair of interleaved rows corresponds to one depth-map row step of 0.5.
    y_pos = (
        torch.arange(n_rows).reshape(-1, 1)
        * torch.ones((1, n_cols))
        * 0.5
    )
    # Sub-pixel offset so ABC and ACD centroids sit at the correct 1/3, 2/3 positions
    y_offset = torch.empty((n_rows, n_cols), dtype=depth_map.dtype)
    y_offset[0::2, :] =  1/3          # ABC centroid
    y_offset[1::2, :] =  2/3 - 0.5   # ACD centroid
    y_pos = y_pos + y_offset

    # --- x (cross-track) centroid positions ---
    x_pos = (
        torch.arange(n_cols).reshape(1, -1)
        * torch.ones((n_rows, 1))
    )
    x_offset = torch.empty((n_rows, n_cols), dtype=depth_map.dtype)
    x_offset[0::2, :] = 1/3
    x_offset[1::2, :] = 2/3
    x_pos = x_pos + x_offset

    # --- leading-edge heights for occlusion (AC / AD edges face the sonar) ---
    # Using the near edge of each triangle rather than the centroid gives a more
    # physically accurate shadow boundary.
    occ_heights_abc = (depth_map[1:, :-1] + depth_map[:-1, 1:]) / 2  # AC edge
    occ_heights_acd = (depth_map[1:, :-1] + depth_map[1:, 1:])  / 2  # AD edge
    occ_heights = torch.empty((n_rows, n_cols), dtype=depth_map.dtype)
    occ_heights[0::2, :] = occ_heights_abc
    occ_heights[1::2, :] = occ_heights_acd

    # --- vector from centroid to sonar ---
    dy = sonar_y - y_pos
    dz_centroid  = sonar_z - mesh_heights   # used for Lambertian / range calc
    dz_occl      = sonar_z - occ_heights    # used for occlusion decision

    # Slant range uses the leading-edge z so occlusion and range are consistent
    slant_range = (dy**2 + dz_occl**2) ** 0.5   # dx = 0 (2-D geometry)

    suv_y = dy     / slant_range
    suv_z = dz_occl / slant_range               # leading-edge unit-vector z

    # Full 3-D unit vector (x-component is zero in this 2-D sonar model)
    sonar_unit_vec = torch.stack(
        (torch.zeros(suv_z.shape), suv_y, suv_z), dim=0
    )

    return sonar_unit_vec, suv_z, slant_range


# ---------------------------------------------------------------------------
# Helper: differentiable occlusion mask
# ---------------------------------------------------------------------------

def _compute_occlusion_mask(
    suv_z: torch.Tensor,
    sigmoid_sharpness: float = 1000.0,
) -> torch.Tensor:
    """
    Compute a per-triangle occlusion multiplier in [0, 1].

    A triangle is occluded (multiplier → 0) when another triangle between it
    and the sonar has a steeper vertical angle to the sonar — i.e. intercepts
    the acoustic ray first.

    Method
    ------
    1. ``suv_z`` is the z-component of the unit vector from each triangle to
       the sonar.  Scanning from the far end (sonar side) toward the near end,
       a triangle is visible only if its ``suv_z`` is at least as large as the
       minimum ``suv_z`` seen so far from the sonar side.

    2. A cumulative minimum along the range dimension captures this: any
       triangle whose ``suv_z`` exceeds the running minimum is in shadow.

    3. The hard step is replaced by a sigmoid of sharpness ``A`` to keep the
       computation differentiable.  Two slightly shifted versions are averaged
       to smooth the shadow boundary and suppress a "flicker" artifact that
       appears when A is large.

    Parameters
    ----------
    suv_z             : (n_rows, n_cols) z-component of sonar unit vector
    sigmoid_sharpness : steepness of the sigmoid approximation (higher → harder
                        shadow edge, but gradient signal becomes sparse)

    Returns
    -------
    occlusion_mask : (n_rows, n_cols) values in [0, 1]; 1 = fully visible
    """
    A = sigmoid_sharpness

    # Cumulative minimum of suv_z scanning from the sonar end (high row index)
    # toward the near end (low row index).  At each position this gives the
    # most "upward-facing" angle seen between here and the sonar.
    min_dz_current = torch.flip(
        torch.cummin(torch.flip(suv_z, dims=(0,)), dim=0)[0],
        dims=(0,),
    )

    # Shift the cummin forward by one row so the shadow starts exactly at the
    # occluding edge, not one step behind it.
    min_dz_next = torch.cat(
        [min_dz_current[1:, :], min_dz_current[0:1, :]], dim=0
    )

    # Sigmoid offset: 4.595 / A places the 99th-percentile of the sigmoid
    # transition at the shadow boundary rather than its midpoint.
    shift = 4.59512 / A

    def _sigmoid_occlusion(suv_z, min_dz):
        """1 where suv_z >= min_dz + shift (visible), 0 where occluded."""
        return 1.0 - (
            1.0 / (1.0 + torch.exp(-A * (suv_z - min_dz - shift)))
        )

    # Average two estimates (straddling the shadow edge) to reduce flicker
    mask_a = _sigmoid_occlusion(suv_z, min_dz_next)
    mask_b = _sigmoid_occlusion(suv_z, min_dz_current)
    return (mask_a + mask_b) / 2.0


# ---------------------------------------------------------------------------
# Helper: range image (sonar waterfall column)
# ---------------------------------------------------------------------------

def _build_range_image(
    slant_range: torch.Tensor,
    intensity_2d: torch.Tensor,
    output_shape: tuple,
    depth_map_shape: tuple,
    sonar_y: float,
    sonar_z: float,
) -> torch.Tensor:
    """
    Smear each triangle's intensity into the output range bins using a
    Gaussian point-spread function along the range dimension.

    Parameters
    ----------
    slant_range    : (n_rows, n_cols) distance from each centroid to the sonar
    intensity_2d   : (n_rows, n_cols) per-triangle intensity before range binning
    output_shape   : (N_range_bins, N_cross_track)
    depth_map_shape: (H, W) — used to compute min/max expected slant ranges
    sonar_y, sonar_z : sonar position (pixel units)

    Returns
    -------
    range_image : (N_range_bins, N_cross_track) accumulated intensity image
    """
    N_ranges = output_shape[0]
    H = depth_map_shape[0]

    # Slant-range limits corresponding to the near and far edges of the swath
    r_near = ((sonar_y - H) ** 2 + sonar_z ** 2) ** 0.5
    r_far  = (sonar_y ** 2 + sonar_z ** 2) ** 0.5

    # Uniformly spaced range bins spanning the full swath
    range_bins = torch.linspace(r_near, r_far, N_ranges)           # (N_ranges,)
    range_bins = range_bins.view(1, 1, -1).expand(
        slant_range.shape[0], slant_range.shape[1], -1
    )                                                               # (n_rows, n_cols, N_ranges)

    # Gaussian PSF: each triangle contributes to all bins weighted by distance
    # from its true slant range.  sigma² = 0.5 is a fixed empirical value.
    sigma2 = 0.5
    delta  = range_bins - slant_range[:, :, None]                  # x - μ
    psf    = (1.0 / (sigma2 * 2 * np.pi) ** 0.5) * torch.exp(
        (-1.0 / (2.0 * sigma2)) * delta ** 2
    )

    # Accumulate: weight each triangle's intensity by its PSF across all bins
    # Result shape: (N_ranges, N_cross_track) after summing over triangles
    range_image = torch.sum(psf * intensity_2d[:, :, None], dim=0)
    return range_image


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def dif_render(
    depth_map: torch.Tensor,
    sonar_range: float,
    sonar_height: float,
    sigmoid_sharpness: float = 100.0,
    use_occlusion: bool = True,
    use_area: bool = True,
    use_lambertian: bool = True,
    use_range_image: bool = True,
    output_render_shape: tuple = (228, 228),
    plot_normals: bool = False,
) -> torch.Tensor:
    """
    Render a differentiable side-scan sonar image from a seabed depth map.

    The renderer models three physical intensity contributions:

    * **Lambertian backscatter** — intensity proportional to the cosine of the
      angle between the surface normal and the sonar direction.
    * **Area weighting** — larger triangle facets return more acoustic energy.
    * **Occlusion (acoustic shadow)** — facets hidden behind taller terrain
      receive no insonification and contribute zero intensity.

    Intensity is then smeared into range bins using a Gaussian PSF to produce
    a realistic waterfall-style sonar image.

    All operations use PyTorch, so gradients propagate through the renderer to
    the input ``depth_map``.

    Parameters
    ----------
    depth_map : torch.Tensor
        2-D (H, W) or batched (1, 1, H, W) seabed height map.
        Pixel spacing is fixed at 0.02 m.
    sonar_range : float
        Ground-plane range from the nadir track to the far edge of the swath,
        in metres.
    sonar_height : float
        Altitude of the sonar above the seabed datum, in metres.
    sigmoid_sharpness : float, optional
        Controls the sharpness of the differentiable shadow boundary.
        Higher values approach a hard step but reduce gradient signal near
        shadow edges.  Default: 100.
    use_occlusion : bool, optional
        Include the occlusion (shadow) multiplier.  Default: True.
    use_area : bool, optional
        Weight intensity by triangle area.  Default: True.
    use_lambertian : bool, optional
        Apply Lambertian (cosine) backscatter weighting.  Default: True.
    use_range_image : bool, optional
        Project intensity into slant-range bins (produces the waterfall image).
        If False, returns the raw 2-D intensity map.  Default: True.
    output_render_shape : tuple of int, optional
        (N_range_bins, N_cross_track) shape of the output image.
        Default: (228, 228).
    plot_normals : bool, optional
        If True, display debug figures for normals, depth map, and intensity.
        Default: False.

    Returns
    -------
    intensity : torch.Tensor
        Rendered sonar image of shape ``output_render_shape``.
    """

    # ------------------------------------------------------------------
    # 0. Input normalisation
    # ------------------------------------------------------------------
    if depth_map.dim() == 4:
        # Accept batched input (1, 1, H, W) — strip batch / channel dims
        depth_map = depth_map[0, 0, :, :]

    # ------------------------------------------------------------------
    # 1. Unit conversion: metres → pixels  (pixel pitch = 0.02 m)
    # ------------------------------------------------------------------
    PIXEL_SIZE = torch.tensor(0.02, requires_grad=False)  # metres per pixel

    # Sonar y-position: ground range puts the sonar behind the far edge of the
    # depth map, so we add the map height to get the sonar row index.
    sonar_y_px = (sonar_range / PIXEL_SIZE) + depth_map.shape[0]
    sonar_z_px = sonar_height / PIXEL_SIZE

    # ------------------------------------------------------------------
    # 2. Mesh geometry (heights, normals, areas)
    # ------------------------------------------------------------------
    mesh_heights, normals, areas = _build_mesh_triangles(depth_map)

    # ------------------------------------------------------------------
    # 3. Sonar unit vectors & slant range per triangle
    # ------------------------------------------------------------------
    sonar_unit_vec, suv_z, slant_range = _build_sonar_unit_vectors(
        depth_map, mesh_heights, sonar_y_px, sonar_z_px
    )

    # ------------------------------------------------------------------
    # 4. Lambertian backscatter coefficient
    #    dot product of surface normal with sonar direction gives cosine
    #    of the incidence angle; clamp negatives (back-faces) to zero.
    # ------------------------------------------------------------------
    lambertian = torch.sum(normals * sonar_unit_vec, dim=0)
    lambertian = lambertian.clamp(min=0.0)

    # ------------------------------------------------------------------
    # 5. Occlusion mask
    # ------------------------------------------------------------------
    occlusion_mask = _compute_occlusion_mask(suv_z, sigmoid_sharpness)

    # ------------------------------------------------------------------
    # 6. Composite intensity
    #    Multiply whichever physical terms are enabled.
    # ------------------------------------------------------------------
    intensity = torch.ones(lambertian.shape)
    if use_area:
        intensity = intensity * areas
    if use_lambertian:
        intensity = intensity * lambertian
    if use_occlusion:
        intensity = intensity * occlusion_mask

    # ------------------------------------------------------------------
    # 7. Range image projection
    #    Each triangle is smeared into slant-range bins via a Gaussian PSF.
    # ------------------------------------------------------------------
    if use_range_image:
        intensity = _build_range_image(
            slant_range, intensity, output_render_shape,
            depth_map.shape, sonar_y_px, sonar_z_px,
        )

    # ------------------------------------------------------------------
    # 8. Post-processing
    #    Pad one row so the output aligns with the depth map, then rotate
    #    so that near range is at the bottom (conventional sonar display).
    # ------------------------------------------------------------------
    # Add a zero row at the top to restore the original row count
    intensity = F.pad(intensity, (0, 0, 1, 0), mode='constant', value=0.0)
    # Rotate 90° so cross-track runs horizontally in the output image
    intensity = torch.rot90(intensity)

    # ------------------------------------------------------------------
    # 9. Optional debug plots
    # ------------------------------------------------------------------
    if plot_normals:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        normals_rgb = normals.detach().numpy().transpose(1, 2, 0).copy()
        # Map z-normals from [-1, 0] → [0, 128] and xy from [-1,1] → [0,255]
        normals_rgb[:, :, 2][normals_rgb[:, :, 2] > 0] = 0
        normals_rgb = np.array((normals_rgb + 1) * (255 / 2), dtype=np.int32)
        ax.imshow(normals_rgb, aspect=0.5)
        ax.set_title("Surface normals (RGB)")

        fig, ax = plt.subplots()
        ax.imshow(depth_map.detach().numpy())
        ax.set_title("Depth map")

        fig, ax = plt.subplots()
        ax.imshow(intensity.detach().numpy())
        ax.set_title("Rendered sonar image")
        plt.show()

    return intensity


if __name__ == "__main__":
    # Simple example
    def half_globe_shape( radius=500):
        size = radius * 2
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        center_x, center_y = size // 2, size // 2
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Clip distances to be within the radius
        distances = np.clip(distances, 0, radius)
        
        hemisphere =  np.sqrt(radius**2 - distances**2)
        return hemisphere
    
    depth_map = np.zeros((224,224))
        
    radius = 40
    out = half_globe_shape(radius)
    cx = 120
    cy = 170
    depth_map[cy-radius:cy+radius,cx-radius:cx+radius] = out
    
    depth_map[150:220,100:150] = 40
    out = dif_render(depth_map=torch.tensor(depth_map),
                                sonar_range=70, 
                                sonar_height=10, 
                                use_occlusion = True,
                                use_area = True,
                                use_lambertian= True,
                                use_range_image = True,
                                plot_normals = True)
    

    