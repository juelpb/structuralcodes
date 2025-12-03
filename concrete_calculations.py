"""Scaffold for rectangular RC section analysis.

This wires together the pieces that exist in the codebase:
- Geometry + reinforcement layout (top/bottom layers on a rectangular
  section) built with shapely-backed geometries.
- Materials from EC2:2004 (B45 concrete by default, configurable steel).
- Section solver via `GenericSection` with Marin/Fiber integrators to get
  bending capacity and moment-curvature.
- Hooks/placeholders for shear stirrups and visualization.

Limitations (left as TODOs):
- No bundled bars helper; stirrup geometry/drawing is not implemented in
  the library, so a stub is provided.
- No automated ULS/SLS/ALS utilisation checks or torsion/shear interaction.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
import tempfile
import uuid

import numpy as np

try:
    import matplotlib
    try:
        matplotlib.use('Agg')  # prefer non-GUI backend to avoid Qt dependency
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    Circle = None
    Rectangle = None

from structuralcodes import set_design_code
from structuralcodes.codes.ec2_2004 import _section_7_3_crack_control as ec2_cr
from structuralcodes.geometry import (RectangularGeometry,
                                      add_reinforcement_line)
from structuralcodes.materials.concrete import create_concrete
from structuralcodes.materials.reinforcement import create_reinforcement
from structuralcodes.sections import (GenericSection,
                                      calculate_elastic_cracked_properties)


def _maybe_show(show: bool, fig=None, save_path: str | None = None):
    """Show or save matplotlib figures when interactive backends are unavailable."""
    if plt is None:
        return
    fig = fig or plt.gcf()
    if fig is None:
        return

    backend = (plt.get_backend() or '').lower()
    target_path = Path(save_path) if save_path else None

    if not show and target_path is None:
        return

    if not show and target_path is not None:
        fig.savefig(target_path, bbox_inches='tight')
        return

    if 'agg' in backend or 'template' in backend or backend == 'agg':
        if target_path is None:
            target_path = Path(tempfile.gettempdir()) / f'plot_{uuid.uuid4().hex}.png'
        fig.savefig(target_path, bbox_inches='tight')
        warnings.warn(
            f"matplotlib backend '{backend}' is non-interactive; saved figure to {target_path}"
        )
        return

    fig.show()


@dataclasses.dataclass
class StirrupLayout:
    """Shear stirrup layout (rendering only, no calculations yet)."""

    diameter: float  # mm
    spacing: float  # mm
    cover_side: float  # mm
    cover_top: float  # mm
    cover_bottom: float  # mm


@dataclasses.dataclass
class RectangularRCInputs:
    """User inputs for a rectangular RC section."""

    width: float
    height: float
    cover_top: float
    cover_bottom: float
    cover_side: float
    bar_diameter_top: float
    bar_diameter_bottom: float
    n_bars_top: int
    n_bars_bottom: int
    mesh_size: float = 0.01  # Fiber integrator mesh density (0-1)
    integrator: str = 'fiber'  # 'fiber' or 'marin'
    bundle: bool = False  # Not supported yet; placeholder only.
    stirrup_diameter: float = 0.0  # mm, optional (for rendering only)
    stirrup_spacing: float = 0.0  # mm, optional (for rendering only)


@dataclasses.dataclass
class LoadState:
    """Basic set of section forces."""

    n: float = 0.0  # axial force in kN (tension +)
    m_y: float = 0.0  # bending about strong axis (y) in kNm; + => tension at bottom
    m_z: float = 0.0  # bending about weak axis (z) in kNm
    v_y: float = 0.0  # shear in kN (not solved here)
    torsion: float = 0.0  # torsion in kNm (not solved here)

    def to_internal_units(self) -> dict[str, float]:
        """Convert kN/kNm inputs to N/Nmm for the solver."""
        return {
            'n': self.n * 1e3,  # kN -> N
            # Flip sign to match solver convention (positive curvature -> tension top).
            'm_y': -self.m_y * 1e6,  # kNm -> Nmm
            'm_z': self.m_z * 1e6,
            'v_y': self.v_y * 1e3,
            'torsion': self.torsion * 1e6,
        }


def _build_rc_geometry(
    cfg: RectangularRCInputs, concrete, reinforcement
) -> RectangularGeometry:
    """Create a rectangular geometry and add top/bottom bar layers."""
    if cfg.bundle:
        raise NotImplementedError('Bundled bars are not supported yet.')

    # Adjust concrete cover if stirrup is present (bars move inward)
    stirrup_cover = cfg.stirrup_diameter if cfg.stirrup_diameter > 0 else 0.0
    cover_top = cfg.cover_top + stirrup_cover
    cover_bottom = cfg.cover_bottom + stirrup_cover
    cover_side = cfg.cover_side + stirrup_cover

    geom = RectangularGeometry(
        width=cfg.width,
        height=cfg.height,
        material=concrete,
        concrete=True,
        origin=(0.0, 0.0),
    )

    # Bottom layer
    y0_bot = -cfg.width / 2 + cover_side + cfg.bar_diameter_bottom / 2
    y1_bot = cfg.width / 2 - cover_side - cfg.bar_diameter_bottom / 2
    z_bot = (
        -cfg.height / 2
        + cover_bottom
        + cfg.bar_diameter_bottom / 2
    )
    geom = add_reinforcement_line(
        geom,
        (y0_bot, z_bot),
        (y1_bot, z_bot),
        cfg.bar_diameter_bottom,
        reinforcement,
        n=cfg.n_bars_bottom,
        group_label='bottom',
    )

    # Top layer
    y0_top = -cfg.width / 2 + cover_side + cfg.bar_diameter_top / 2
    y1_top = cfg.width / 2 - cover_side - cfg.bar_diameter_top / 2
    z_top = cfg.height / 2 - cover_top - cfg.bar_diameter_top / 2
    geom = add_reinforcement_line(
        geom,
        (y0_top, z_top),
        (y1_top, z_top),
        cfg.bar_diameter_top,
        reinforcement,
        n=cfg.n_bars_top,
        group_label='top',
    )

    return geom


def build_section(cfg: RectangularRCInputs) -> GenericSection:
    """Build a GenericSection with EC2 materials and requested integrator."""
    set_design_code('ec2_2004')
    concrete = create_concrete(fck=45)
    reinforcement = create_reinforcement(
        fyk=500, Es=200000, ftk=550, epsuk=0.07
    )

    geom = _build_rc_geometry(cfg, concrete, reinforcement)
    sec = GenericSection(
        geom,
        name='RectangularRC',
        integrator=cfg.integrator.lower(),
        mesh_size=cfg.mesh_size,
    )
    return sec


def solve_bending_and_mc(
    section: GenericSection, load: LoadState
) -> dict[str, object]:
    """Solve bending strength and moment-curvature along bending axis."""
    internal = load.to_internal_units()
    theta = (
        math.atan2(internal['m_z'], internal['m_y'])
        if (load.m_y or load.m_z)
        else 0.0
    )

    bending = section.section_calculator.calculate_bending_strength(
        theta=theta, n=internal['n']
    )
    mc = section.section_calculator.calculate_moment_curvature(
        theta=theta, n=internal['n']
    )
    cracked_props = calculate_elastic_cracked_properties(
        section, theta=theta
    )

    # If a target bending moment is specified, find the closest point on MC curve
    mc_target = None
    target_m_int = internal['m_y']
    if target_m_int and hasattr(mc, 'm_y'):
        moments = mc.m_y
        idx = min(
            range(len(moments)), key=lambda i: abs(moments[i] - target_m_int)
        )
        mc_target = {
            'target_m_y_kNm': -target_m_int / 1e6,
            'closest_m_y_kNm': -moments[idx] / 1e6,
            'chi_y': mc.chi_y[idx],
            'chi_z': mc.chi_z[idx],
            'eps_a': mc.eps_a[idx] if hasattr(mc, 'eps_a') else None,
            'index': idx,
        }

    return {
        'theta': theta,
        'bending_strength': bending,
        'bending_strength_kNm': {
            # Convert back to user convention: +M_y -> tension bottom
            'm_y': -bending.m_y / 1e6,
            'm_z': bending.m_z / 1e6,
            'n': bending.n / 1e3,
        },
        'moment_curvature': mc,
        'cracked_props': cracked_props,
        'mc_at_target_m': mc_target,
    }


def solve_section_for_load(
    section: GenericSection,
    load: LoadState,
    tol: float = 1e-2,
    max_iter: int = 50,
) -> dict[str, object]:
    """Find strain that equilibrates the given loads (N, My, Mz).

    Uses Newton-Raphson on the section stiffness (from integrator).
    """
    internal = load.to_internal_units()
    target = np.array([internal['n'], internal['m_y'], internal['m_z']])
    strain = np.zeros(3)

    for _ in range(max_iter):
        n_int, my_int, mz_int = section.section_calculator.integrate_strain_profile(
            strain
        )
        res = target - np.array([n_int, my_int, mz_int])
        if np.max(np.abs(res)) <= tol:
            break
        k_sec = section.section_calculator.integrate_strain_profile(
            strain, integrate='modulus'
        )
        try:
            delta = np.linalg.solve(k_sec, res)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Section stiffness is singular.') from exc
        strain += delta
    else:
        raise ValueError('Did not converge to the applied loads.')

    n_int, my_int, mz_int = section.section_calculator.integrate_strain_profile(
        strain
    )

    bending_ns = SimpleNamespace(
        eps_a=float(strain[0]),
        chi_y=float(strain[1]),
        chi_z=float(strain[2]),
    )
    steel = compute_steel_stresses(section, bending_ns)
    return {
        'strain': strain,
        'internal': (n_int, my_int, mz_int),
        'target': target,
        'residual': target - np.array([n_int, my_int, mz_int]),
        'steel_stresses': steel,
        'strain_result': bending_ns,
    }


def compute_steel_stresses(
    section: GenericSection,
    strain_result,
) -> list[dict[str, float]]:
    """Compute steel stresses (MPa) for each bar given a bending result."""
    eps_a = strain_result.eps_a
    chi_y = strain_result.chi_y
    chi_z = strain_result.chi_z
    bar_data = []
    for bar in section.geometry.point_geometries:
        eps_bar = eps_a + chi_y * bar.y - chi_z * bar.x
        sigma_bar = bar.material.constitutive_law.get_stress(eps_bar)
        bar_data.append(
            {
                'x_mm': bar.x,
                'y_mm': bar.y,
                'strain': eps_bar,
                'stress_mpa': sigma_bar,
            }
        )
    return bar_data


def _collect_side_data(section: GenericSection, side: str, strain_result):
    """Collect bar data and tension status for top/bottom."""
    bars = [
        bar for bar in section.geometry.point_geometries if bar.group_label == side
    ]
    if not bars:
        return None
    maxx, maxy = section.geometry.geometries[0].polygon.bounds[2:]
    minx, miny = section.geometry.geometries[0].polygon.bounds[:2]
    eps_a = strain_result.eps_a
    chi_y = strain_result.chi_y
    chi_z = strain_result.chi_z

    bar_info = []
    for bar in bars:
        eps_bar = eps_a + chi_y * bar.y - chi_z * bar.x
        sigma_bar = bar.material.constitutive_law.get_stress(eps_bar)
        bar_info.append(
            {
                'bar': bar,
                'eps': eps_bar,
                'sigma': sigma_bar,
            }
        )

    # Determine cover to tension face
    if side == 'top':
        cover = maxy - max(b['bar'].y for b in bar_info) - bars[0].diameter / 2
        fiber_strain = eps_a + chi_y * maxy
    else:
        cover = min(b['bar'].y for b in bar_info) - miny - bars[0].diameter / 2
        fiber_strain = eps_a + chi_y * miny

    return {
        'bars': bar_info,
        'cover': cover,
        'fiber_strain': fiber_strain,
        'width': maxx - minx,
        'height': maxy - miny,
        'maxz': maxy,
        'minz': miny,
    }


def compute_crack_widths_ec2(
    section: GenericSection,
    strain_result,
    steel_stresses=None,
    load_type: str = 'short',
) -> dict[str, float]:
    """Compute crack width at top and bottom using EC2:2004 helpers (simplified).

    Returns mm values for 'top' and 'bottom' where tension occurs; otherwise None.
    """
    if steel_stresses is None:
        steel_stresses = compute_steel_stresses(section, strain_result)
    steel_lookup = {(bar['x_mm'], bar['y_mm']): bar for bar in steel_stresses}

    def side_result(side: str):
        data = _collect_side_data(section, side, strain_result)
        if data is None:
            return None
        cover = max(data['cover'], 0.0)
        if data['fiber_strain'] <= 0:
            return None  # compression side
        bars = data['bars']
        phi = bars[0]['bar'].diameter
        # spacing along width (y-direction)
        y_coords = sorted(b['bar'].x for b in bars)
        spacing = max(
            (y_coords[i + 1] - y_coords[i]) for i in range(len(y_coords) - 1)
        ) if len(y_coords) > 1 else 0.0

        As_tot = sum(math.pi * b['bar'].diameter ** 2 / 4 for b in bars)
        width = data['width']
        height = data['height']
        chi_y = strain_result.chi_y
        eps_a = strain_result.eps_a
        z_na = None if chi_y == 0 else -eps_a / chi_y
        maxz = data['maxz']
        minz = data['minz']
        if z_na is None:
            tension_depth = height / 2
        elif side == 'top':
            tension_depth = max(maxz - min(z_na, maxz), 0.0)
        else:
            tension_depth = max(max(z_na, minz) - minz, 0.0)
        Ac_eff = width * max(tension_depth, 1e-6)
        rho_p_eff = As_tot / Ac_eff if Ac_eff > 0 else 0
        if rho_p_eff <= 0:
            return None

        k1 = ec2_cr.k1('bond')
        k2 = 0.5  # bending
        kt = ec2_cr.kt(load_type)
        concrete = section.geometry.geometries[0].material
        fct_eff = getattr(concrete, 'fctm', 2.9)
        Es = bars[0]['bar'].material.constitutive_law.get_tangent(eps=0)
        Ecm = getattr(concrete, 'Ecm', 30000)
        alpha_e = Es / Ecm

        spacing_threshold = ec2_cr.w_spacing(cover, phi)
        if spacing == 0:
            spacing = spacing_threshold  # single bar
        if spacing <= spacing_threshold:
            sr_max = ec2_cr.sr_max_close(cover, phi, rho_p_eff, k1, k2)
        else:
            x = height - tension_depth
            sr_max = ec2_cr.sr_max_far(height, x)

        sigma_s = max(b['sigma'] for b in bars if b['sigma'] > 0) if bars else 0
        if sigma_s <= 0:
            return None
        eps_diff = ec2_cr.eps_sm_eps_cm(
            sigma_s=sigma_s,
            alpha_e=alpha_e,
            rho_p_eff=rho_p_eff,
            kt=kt,
            fct_eff=fct_eff,
            Es=Es,
        )
        return ec2_cr.wk(sr_max, eps_diff)

    return {'top': side_result('top'), 'bottom': side_result('bottom')}


def plot_strain_field(
    section: GenericSection,
    strain_result,
    show: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """Plot the strain field over the concrete polygon (scatter mask)."""
    if plt is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'matplotlib is required for plotting; install it to enable.'
        )

    from shapely.geometry import Point

    poly = section.geometry.geometries[0].polygon
    minx, miny, maxx, maxy = poly.bounds
    gx = int(max(20, (maxx - minx) / 10))
    gy = int(max(20, (maxy - miny) / 10))
    xs = [minx + i * (maxx - minx) / gx for i in range(gx + 1)]
    ys = [miny + j * (maxy - miny) / gy for j in range(gy + 1)]

    eps_a = strain_result.eps_a
    chi_y = strain_result.chi_y
    chi_z = strain_result.chi_z

    points_x, points_y, strains = [], [], []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if not poly.contains(p):
                continue
            eps = eps_a + chi_y * y - chi_z * x
            points_x.append(x)
            points_y.append(y)
            strains.append(eps)

    if ax is None:
        _, ax = plt.subplots()
    sc = ax.scatter(points_x, points_y, c=strains, cmap='coolwarm', s=12)
    plt.colorbar(sc, ax=ax, label='strain [-]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    ax.grid(True, linestyle=':')
    _maybe_show(show, fig=ax.figure if ax is not None else None, save_path=save_path)
    return ax


def plot_reinf_stress(
    section: GenericSection,
    strain_result,
    show: bool = True,
    ax=None,
    steel_stresses=None,
    save_path: str | None = None,
):
    """Plot reinforcement stress as colored circles."""
    if plt is None or Circle is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'matplotlib is required for plotting; install it to enable.'
        )

    if ax is None:
        _, ax = plt.subplots()

    stresses = steel_stresses or compute_steel_stresses(section, strain_result)
    for geo in section.geometry.geometries:
        x, y = geo.polygon.exterior.xy
        ax.fill(x, y, alpha=0.15, facecolor='#cccccc', edgecolor='#444444')

    if stresses:
        min_s = min(item['stress_mpa'] for item in stresses)
        max_s = max(item['stress_mpa'] for item in stresses)
    else:
        min_s = max_s = 0.0

    for bar, data in zip(section.geometry.point_geometries, stresses):
        if max_s != min_s:
            norm = (data['stress_mpa'] - min_s) / (max_s - min_s)
        else:
            norm = 0.5
        color = plt.cm.viridis(norm)
        circ = Circle(
            (bar.x, bar.y),
            radius=bar.diameter / 2,
            facecolor=color,
            edgecolor='#222222',
            alpha=0.9,
        )
        ax.add_patch(circ)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    _maybe_show(show, fig=ax.figure if ax is not None else None, save_path=save_path)
    return ax


def stirrup_layout_placeholder():
    """Reserved for future shear stirrup checks (kept for reference)."""
    raise NotImplementedError(
        'Shear stirrup generation/drawing is not implemented in this '
        'codebase. Use StirrupLayout + render_section_diagram for visuals.'
    )


def _render_shear_stirrup(ax, geom: RectangularGeometry, layout: StirrupLayout):
    """Draw a closed stirrup line for visual control."""
    if layout.diameter <= 0:
        return
    if Rectangle is None:
        return
    minx, miny, maxx, maxy = geom.polygon.bounds
    d = layout.diameter
    # Outer edge of stirrup (cover is to the outer surface of the bar)
    outer_minx = minx + layout.cover_side
    outer_maxx = maxx - layout.cover_side
    outer_miny = miny + layout.cover_bottom
    outer_maxy = maxy - layout.cover_top
    width_outer = outer_maxx - outer_minx
    height_outer = outer_maxy - outer_miny
    # Inner edge (subtract bar thickness)
    inner_minx = outer_minx + d
    inner_maxx = outer_maxx - d
    inner_miny = outer_miny + d
    inner_maxy = outer_maxy - d
    width_inner = max(inner_maxx - inner_minx, 0)
    height_inner = max(inner_maxy - inner_miny, 0)

    # Draw outer filled rectangle then punch inner hole with background color
    outer_rect = Rectangle(
        (outer_minx, outer_miny),
        width_outer,
        height_outer,
        linewidth=0,
        edgecolor=None,
        facecolor='#2c7bb6',
        alpha=0.35,
        label='stirrup',
    )
    ax.add_patch(outer_rect)
    if width_inner > 0 and height_inner > 0:
        inner_rect = Rectangle(
            (inner_minx, inner_miny),
            width_inner,
            height_inner,
            linewidth=0,
            edgecolor=None,
            facecolor='white',
            alpha=1.0,
        )
        ax.add_patch(inner_rect)


def render_cross_section(
    section: GenericSection,
    show: bool = True,
    ax=None,
    stirrup: StirrupLayout | None = None,
    save_path: str | None = None,
):
    """Render the cross section with concrete, bars, and optional stirrup."""
    if plt is None or Circle is None or Rectangle is None:  # pragma: no cover
        raise RuntimeError(
            'matplotlib is required for rendering; install it to enable '
            'visual checks.'
        )

    if ax is None:
        _, ax = plt.subplots()

    for geo in section.geometry.geometries:
        x, y = geo.polygon.exterior.xy
        ax.fill(
            x,
            y,
            alpha=0.25,
            facecolor='#e0e0e0',
            edgecolor='#444444',
            label=geo.name,
        )
        if stirrup is not None:
            _render_shear_stirrup(ax, geo, stirrup)

    for bar in section.geometry.point_geometries:
        circ = Circle(
            (bar.x, bar.y),
            radius=bar.diameter / 2,
            facecolor='#d62728',
            edgecolor='#8c1f1f',
            alpha=0.85,
        )
        ax.add_patch(circ)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')

    _maybe_show(show, fig=ax.figure if ax is not None else None, save_path=save_path)
    return ax


def render_section_diagram(
    section: GenericSection,
    strain_result,
    stirrup: StirrupLayout | None = None,
    steel_stresses=None,
    show: bool = True,
    save_path: str | None = None,
):
    """Render a three-part diagram: section, strain, stress (absolute scales).

    Args:
        section: GenericSection instance.
        strain_result: object with eps_a, chi_y, chi_z (e.g. from equilibrium
            solve or bending strength).
        stirrup: optional StirrupLayout for drawing.
        steel_stresses: optional precomputed bar stresses; if None, they are
            computed from strain_result.
    """
    if plt is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'matplotlib is required for rendering; install it to enable.'
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (i) Section
    render_cross_section(section, show=False, ax=axes[0], stirrup=stirrup)
    axes[0].set_title('Section')
    axes[0].legend(loc='upper right', fontsize=8)

    # geometry bounds
    poly = section.geometry.geometries[0].polygon
    minx, miny, maxx, maxy = poly.bounds

    # (ii) Strain diagram along z (vertical)
    eps_a = strain_result.eps_a
    chi_y = strain_result.chi_y
    eps_top = eps_a + chi_y * maxy
    eps_bot = eps_a + chi_y * miny
    z_na = None if chi_y == 0 else -eps_a / chi_y
    axes[1].plot([eps_bot, eps_top], [miny, maxy], color='#1f77b4')
    axes[1].scatter([eps_bot, eps_top], [miny, maxy], color='#1f77b4', s=30)
    axes[1].axvline(0, color='#999999', linestyle='--', linewidth=1)
    if z_na is not None and miny <= z_na <= maxy:
        axes[1].axhline(z_na, color='#ff7f0e', linestyle='--', linewidth=1)
        axes[1].text(eps_top, z_na, 'NA', va='bottom', ha='left', fontsize=8)
        axes[1].text(
            eps_bot,
            miny,
            'tension' if eps_bot > 0 else 'compression',
            va='bottom',
            ha='left',
            fontsize=8,
            color='#444444',
        )
        axes[1].text(
            eps_top,
            maxy,
            'tension' if eps_top > 0 else 'compression',
            va='bottom',
            ha='right',
            fontsize=8,
            color='#444444',
        )
    axes[1].set_xlabel('strain [-]')
    axes[1].set_ylabel('z [mm]')
    axes[1].set_title('Strain diagram')
    axes[1].grid(True, linestyle=':')

    # (iii) Stress view with absolute steel stresses (MPa)
    ax3 = axes[2]
    stresses = steel_stresses or compute_steel_stresses(section, strain_result)
    if stresses:
        min_s = min(s['stress_mpa'] for s in stresses)
        max_s = max(s['stress_mpa'] for s in stresses)
        pad = max(50.0, 0.05 * max(abs(min_s), abs(max_s)))
    else:
        min_s = -1.0
        max_s = 1.0
        pad = 1.0

    if z_na is not None:
        if eps_top > 0:
            ax3.fill_betweenx(
                [z_na, maxy],
                0,
                max_s + pad,
                color='#cccccc',
                alpha=0.4,
                label='Concrete compression',
            )
        if eps_bot < 0:
            ax3.fill_betweenx(
                [miny, z_na],
                min_s - pad,
                0,
                color='#e0e0e0',
                alpha=0.2,
                label='Concrete tension',
            )

    for bar_data in stresses:
        color = '#d62728' if bar_data['stress_mpa'] >= 0 else '#1f77b4'
        ax3.scatter(
            bar_data['stress_mpa'],
            bar_data['y_mm'],
            color=color,
            s=20,
            zorder=3,
        )
        ax3.arrow(
            0,
            bar_data['y_mm'],
            bar_data['stress_mpa'],
            0,
            head_width=10,
            head_length=max(2.0, 0.02 * abs(bar_data['stress_mpa'])),
            length_includes_head=True,
            color=color,
        )
    if z_na is not None:
        ax3.axhline(z_na, color='#ff7f0e', linestyle='--', linewidth=1)
    ax3.set_ylim(miny, maxy)
    ax3.set_xlim(min_s - pad, max_s + pad)
    ax3.set_xlabel('stress [MPa]')
    ax3.set_ylabel('z [mm]')
    ax3.set_title('Stress')
    ax3.grid(True, linestyle=':')

    fig.tight_layout()
    _maybe_show(show, fig=fig, save_path=save_path)
    return axes


def render_fiber_mesh(
    section: GenericSection,
    mesh_size: float | None = None,
    show: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """Render the fiber mesh (triangulation) for the section geometry."""
    if plt is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'matplotlib is required for rendering; install it to enable.'
        )
    try:
        import triangle
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'triangle package is required for mesh rendering.'
        ) from exc

    from structuralcodes.sections.section_integrators._fiber_integrator import \
        FiberIntegrator

    if ax is None:
        _, ax = plt.subplots()

    integ = FiberIntegrator()
    ms = mesh_size or getattr(section.section_calculator, 'mesh_size', 0.01)

    for geo in section.geometry.geometries:
        tri_input = integ.prepare_triangulation(geo)
        max_area = geo.area * ms
        mesh = triangle.triangulate(tri_input, f'pq{30:.1f}Aa{max_area}o1')
        verts = mesh['vertices']
        tris = mesh['triangles']
        for tr in tris:
            x = [verts[i][0] for i in tr] + [verts[tr[0]][0]]
            y = [verts[i][1] for i in tr] + [verts[tr[0]][1]]
            ax.plot(x, y, color='#888888', linewidth=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title('Fiber mesh')
    _maybe_show(show, fig=ax.figure if ax is not None else None, save_path=save_path)
    return ax


def example():
    """Minimal end-to-end run; adjust inputs as needed."""
    cfg = RectangularRCInputs(
        width=500,
        height=1000,
        cover_top=50,
        cover_bottom=65,
        cover_side=50,
        bar_diameter_top=25,
        bar_diameter_bottom=25,
        n_bars_top=5,
        n_bars_bottom=5,
        mesh_size=0.005,
        stirrup_diameter=20,
        stirrup_spacing=150,
    )
    load = LoadState(m_y=200)

    section = build_section(cfg)
    results = solve_bending_and_mc(section, load)
    equilibrium = solve_section_for_load(section, load)
    steel = equilibrium['steel_stresses']
    cracks = compute_crack_widths_ec2(
        section, equilibrium['strain_result'], steel_stresses=steel
    )
    stirrup = StirrupLayout(
        diameter=cfg.stirrup_diameter,
        spacing=cfg.stirrup_spacing,
        cover_side=cfg.cover_side,
        cover_top=cfg.cover_top,
        cover_bottom=cfg.cover_bottom,
    )

    # Optional: visualize
    # render_cross_section(section, stirrup=stirrup)
    # plot_strain_field(section, results['bending_strength'])
    # plot_reinf_stress(section, results['bending_strength'])
    render_section_diagram(
        section,
        equilibrium['strain_result'],
        steel_stresses=equilibrium['steel_stresses'],
        stirrup=stirrup,
        show=False, 
        save_path="section_diagram.png"
    )
    render_fiber_mesh(section, show=False, save_path="fiber_mesh.png")

    return section, results, steel, equilibrium, cracks


if __name__ == '__main__':  # pragma: no cover
    section, results, steel, equilibrium, cracks = example()
    bend = results['bending_strength']
    mc = results['moment_curvature']
    bend_kNm = results['bending_strength_kNm']
    print('Bending strength My [kNm]:', bend_kNm['m_y'])
    print('Bending strength Mz [kNm]:', bend_kNm['m_z'])
    print('Axial force [kN]:', bend_kNm['n'])
    print('Moment-curvature points:', len(mc.chi_y))
    print(
        'Closest MC to target My [kNm]:',
        (results['mc_at_target_m'] or {}).get('closest_m_y_kNm')
        if results['mc_at_target_m']
        else None,
    )
    # Applied load equilibrium results
    eq = equilibrium
    print('Applied load residual (N, Nmm, Nmm):', eq['residual'])
    print(
        'First bar stress under applied load [MPa]:',
        eq['steel_stresses'][0]['stress_mpa'] if eq['steel_stresses'] else None,
    )
    print('Crack width top [mm]:', cracks.get('top'))
    print('Crack width bottom [mm]:', cracks.get('bottom'))
