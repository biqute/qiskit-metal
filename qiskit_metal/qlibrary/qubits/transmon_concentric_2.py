# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
# Adapted 2025 by [Your Name]:
#   • Two concentric pads (inner washer/disk + outer annular wedge)
#   • Circular ground‐pocket cutout
#   • Single Josephson junction (straight bridge) connecting inner→outer pads
#   • N “connection pads,” each an annular wedge (arc) with its own pin, 
#     using exactly the same geometry rules as the old resonator.
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE.txt in the root directory or:
#   http://www.apache.org/licenses/LICENSE-2.0
# Any modifications must retain this notice.
"""
TransmonConcentricTwoPadsWithArcConnections

A Qiskit‐Metal BaseQubit subclass that draws exactly:

  1. Inner pad: a washer or disk between radii [ hole_r, hole_r + ip_w ].
  2. Outer pad: an annular wedge between radii [ r_op_inner, r_op_outer ],
     spanning `coverage` degrees (centered at 90°), with rounded semicircular caps.
  3. A single Josephson junction: a straight line from (0, –r_in_outer) → (0, –r_op_inner),
     with width = jj_w.
  4. Circular ground pocket: disk of radius = pocket_r (subtracted on layer 1).
  5. N connection pads: each is an annular wedge (“arc”) between [r_conn_inner, r_conn_outer]
     spanning `coverage` degrees around `angle` (all in “raw” coordinates), with semicircular caps. 
     Each pad has its own QPin running radially through its center angle.  
     (If you still want a “readout” resonator, simply add a pad named `"readout"` in `connection_pads` 
     with the appropriate parameters.)
  6. All geometry is first built around (0,0), then rotated by p.orientation, then translated 
     by (p.pos_x, p.pos_y).
"""

from math import cos, sin, pi
import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import BaseQubit


class TransmonConcentric(BaseQubit):
    """Two‐pad + single JJ + circular pocket + N annular‐wedge connection pads."""

    default_options = Dict(
        # ─── (1) Inner pad + hole ────────────────────────────────
        hole_r       = '0um',      # Radius of central hole. If 0 → inner pad is solid disk of radius ip_w
        ip_w         = '115um',    # Inner pad radial thickness → outer radius_inner = hole_r + ip_w

        # ─── (2) Outer pad parameters ─────────────────────────────
        gap          = '20um',     # Gap between inner and outer pad
        op_w         = '50um',     # Outer pad radial thickness → r_op_outer = r_op_inner + op_w
        coverage     = 360.0,      # Degrees of coverage for outer pad (if 360°→full annulus)

        # ─── (3) Josephson junction ───────────────────────────────
        jj_w         = '5um',      # Thickness of the JJ line
                                  # Junction will run from (0,–r_in_outer) → (0,–r_op_inner)

        # ─── (4) Circular ground pocket ───────────────────────────
        pocket_r     = '500um',    # Radius of circular ground pocket (cutout on layer 1)

        # ─── (5) Inherited CPW / layer (flux removed)─────────────
        cpw_width    = '10um',     # (Unused here, kept for backward compatibility)
        layer        = '1',        # Metal layer for all pads + JJ + connection‐arcs

        # ─── (6) Connection‐pad setup (each pad is an annular wedge + QPin) ───────────────────
        connection_pads = Dict(
            # Example entry for a “readout” pad:
            readout = Dict(
                gap        = '20um',
                width      = '25um',
                coverage   = 120.0,
                angle      =  45.0,
                cpw_length = '80um',
                cpw_width  = '8um',
                cpw_gap    = '4um',
                pad_gap    = '4um', 
            )
        )
    )

    TOOLTIP = ("Two‐pad concentric + single JJ + circular pocket + "
               "arbitrary annular‐wedge connection pads")

    def make(self):
        """Convert parsed options into QGeometry (core qubit + N annular‐wedge pads)."""
        p = self.parse_options()

        # -------------------------------------------------------------------------
        # 1) INNER PAD: washer or disk between [hole_r, hole_r + ip_w]
        # -------------------------------------------------------------------------
        r_hole     = p.hole_r                        # radius of central hole
        r_in_outer = p.hole_r + p.ip_w               # outer radius of inner pad

        if r_hole > 0:
            circ_big_inner  = draw.Point(0, 0).buffer(r_in_outer)
            circ_hole_inner = draw.Point(0, 0).buffer(r_hole)
            inner_pad       = draw.subtract(circ_big_inner, circ_hole_inner)
        else:
            inner_pad = draw.Point(0, 0).buffer(r_in_outer)

        # -------------------------------------------------------------------------
        # 2) OUTER PAD: annular region between [r_op_inner, r_op_outer], coverage°, 
        #    with semicircular caps unless coverage = 360°.
        # -------------------------------------------------------------------------
        r_op_inner = r_in_outer + p.gap
        r_op_outer = r_op_inner + p.op_w
        cov_outer  = float(p.coverage)

        if np.isclose(cov_outer, 360.0):
            circ_big_outer  = draw.Point(0, 0).buffer(r_op_outer)
            circ_hole_outer = draw.Point(0, 0).buffer(r_op_inner)
            outer_pad       = draw.subtract(circ_big_outer, circ_hole_outer)
        else:
            w_op     = r_op_outer - r_op_inner
            r_mid_op = 0.5 * (r_op_outer + r_op_inner)
            θ_start  = 90.0 - cov_outer / 2.0
            θ_end    = 90.0 + cov_outer / 2.0
            N        = 120  # sampling resolution

            # (a) Sample outer circular arc
            thetas_out = np.linspace(θ_start, θ_end, N + 1)
            outer_pts  = [
                (r_op_outer * np.cos(np.deg2rad(th)),
                 r_op_outer * np.sin(np.deg2rad(th)))
                for th in thetas_out
            ]
            # (b) Endcap at θ_end: semicircle radius = w_op/2
            center_end = (
                r_mid_op * np.cos(np.deg2rad(θ_end)),
                r_mid_op * np.sin(np.deg2rad(θ_end))
            )
            phis_end = np.linspace(0, np.pi, N + 1)
            cap_end_pts = [
                (
                    center_end[0] + (w_op / 2) * np.cos(np.deg2rad(θ_end) + φ),
                    center_end[1] + (w_op / 2) * np.sin(np.deg2rad(θ_end) + φ)
                )
                for φ in phis_end
            ]
            # (c) Sample inner arc from θ_end → θ_start
            thetas_in = np.linspace(θ_end, θ_start, N + 1)
            inner_pts = [
                (r_op_inner * np.cos(np.deg2rad(th)),
                 r_op_inner * np.sin(np.deg2rad(th)))
                for th in thetas_in
            ]
            # (d) Startcap at θ_start
            center_start = (
                r_mid_op * np.cos(np.deg2rad(θ_start)),
                r_mid_op * np.sin(np.deg2rad(θ_start))
            )
            phis_start = np.linspace(np.pi, 2 * np.pi, N + 1)
            cap_start_pts = [
                (
                    center_start[0] + (w_op / 2) * np.cos(np.deg2rad(θ_start) + φ),
                    center_start[1] + (w_op / 2) * np.sin(np.deg2rad(θ_start) + φ)
                )
                for φ in phis_start
            ]
            coords_outer = outer_pts + cap_end_pts + inner_pts + cap_start_pts
            outer_pad    = draw.Polygon(coords_outer)

        # -------------------------------------------------------------------------
        # 3) JOSEPHSON JUNCTION: straight line from (0, –r_in_outer) → (0, –r_op_inner).
        # -------------------------------------------------------------------------
        jj_line = draw.LineString([
            (0.0, r_in_outer),
            (0.0, r_op_inner)
        ])

        # -------------------------------------------------------------------------
        # 4) CIRCULAR GROUND POCKET (cutout on layer 1)
        # -------------------------------------------------------------------------
        pocket_thing = draw.Point(0, 0).buffer(p.pocket_r)

        # -------------------------------------------------------------------------
        # 5) ROTATE + TRANSLATE CORE SHAPES (inner_pad, outer_pad, jj_line, pocket_thing)
        # -------------------------------------------------------------------------
        core_shapes = [inner_pad, outer_pad, jj_line, pocket_thing]
        # (a) Rotate about (0,0)
        core_shapes = draw.rotate(core_shapes, p.orientation, origin=(0, 0))
        # (b) Translate by (pos_x, pos_y)
        core_shapes = draw.translate(core_shapes, xoff=p.pos_x, yoff=p.pos_y)
        # Unpack
        inner_pad, outer_pad, jj_line, pocket_thing = core_shapes

        # -------------------------------------------------------------------------
        # 6) ADD QGEOMETRY FOR CORE QUBIT
        # -------------------------------------------------------------------------
        self.add_qgeometry(
            'poly',
            {'poly_inner': inner_pad},
            layer=p.layer,
            subtract=False
        )
        self.add_qgeometry(
            'poly',
            {'poly_outer': outer_pad},
            layer=p.layer,
            subtract=False
        )
        self.add_qgeometry(
            'junction',
            {'poly_jj': jj_line},
            layer=p.layer,
            subtract=False,
            width=p.jj_w
        )
        self.add_qgeometry(
            'poly',
            {'poly_pocket': pocket_thing},
            layer='1',
            subtract=True
        )

        # -------------------------------------------------------------------------
        # 7) NOW CREATE ALL CONNECTION PADS (each one an annular wedge + QPin)
        # -------------------------------------------------------------------------
        self.make_connection_pads()


    # ────────────────────────────────────────────────────────────────────────────
    def make_connection_pads(self):
        """Loop over p.connection_pads and call make_connection_pad(name)."""
        p = self.p
        for name in self.options.connection_pads:
            self.make_connection_pad(name)


    # ────────────────────────────────────────────────────────────────────────────
    def make_connection_pad(self, name: str):
        """Create a single annular‐wedge connection pad named `name`.  

        Each pad’s options must supply:
          • gap      (string, μm)      radial gap from r_op_outer → r_conn_inner  
          • width    (string, μm)      radial thickness → r_conn_outer = r_conn_inner + width  
          • coverage (float, degrees)  angular span (Δθ) of the wedge  
          • angle    (float, degrees)  center angle θ₀ of that wedge (before global rotation)
          • cpw_length : string → μm  (how far the CPW extends outward from r_conn_outer)
          • cpw_width  : string → μm  (CPW center‐trace width)
          • cpw_gap    : string → μm  (dielectric gap on each side of CPW)
    
        Then we build an annular wedge [r_conn_inner→r_conn_outer] spanning [θ₀ − Δθ/2, θ₀ + Δθ/2], 
        with semicircular caps at each angular edge.  Finally we place a QPin at angle θ₀, 
        between r_conn_inner and r_conn_outer.
        """
        p  = self.p
        pc = self.p.connection_pads[name]

        # ─────────────── Parse the pad’s parameters ───────────────
        gap_conn   = pc.gap            # string → float μm
        w_conn     = pc.width          # string → float μm
        cov_conn   = float(pc.coverage) # degrees
        θ_center   = float(pc.angle)    # degrees
        θ0         = θ_center - cov_conn/2.0
        θ1         = θ_center + cov_conn/2.0
        cpw_len     = pc.cpw_length     # string → float μm
        cpw_w       = pc.cpw_width      # string → float μm
        cpw_g       = pc.cpw_gap        # string → float μm
        pad_g       = pc.pad_gap        # string → float μm

        # ─────────────── Compute radii for this pad ───────────────
        # Recompute r_op_outer (in case someone changed ip_w, gap, or op_w externally):
        r_in        = p.hole_r + p.ip_w
        r_op_in     = r_in + p.gap
        r_op_out    = r_op_in + p.op_w

        r_conn_inner = r_op_out + gap_conn
        r_conn_outer = r_conn_inner + w_conn
        w_ann        = r_conn_outer - r_conn_inner
        r_mid_ann    = 0.5 * (r_conn_outer + r_conn_inner)

        #    The gap extends pad_g inside and pad_g outside the metal wedge.
        r_gap_inner  = r_conn_inner - pad_g
        r_gap_outer  = r_conn_outer + pad_g

        # ─────────────── Build the annular‐wedge “pad” in RAW coords ───────────────
        if np.isclose(cov_conn, 360.0):
            # Full annulus from r_conn_inner → r_conn_outer
            circ_big_conn  = draw.Point(0, 0).buffer(r_conn_outer)
            circ_hole_conn = draw.Point(0, 0).buffer(r_conn_inner)
            conn_pad       = draw.subtract(circ_big_conn, circ_hole_conn)
        else:
            N = 120  # sampling resolution

            # (a) Sample outer circular arc from θ0 → θ1 at radius = r_conn_outer
            thetas_out = np.linspace(θ0, θ1, N + 1)
            outer_pts  = [
                (r_conn_outer * np.cos(np.deg2rad(th)),
                 r_conn_outer * np.sin(np.deg2rad(th)))
                for th in thetas_out
            ]

            # (b) Endcap at θ1: semicircle of radius = w_ann/2, center = (r_mid_ann, θ1)
            center_end = (
                r_mid_ann * np.cos(np.deg2rad(θ1)),
                r_mid_ann * np.sin(np.deg2rad(θ1))
            )
            phis_end = np.linspace(0, np.pi, N + 1)
            cap_end_pts = [
                (
                    center_end[0] + (w_ann / 2) * np.cos(np.deg2rad(θ1) + φ),
                    center_end[1] + (w_ann / 2) * np.sin(np.deg2rad(θ1) + φ)
                )
                for φ in phis_end
            ]

            # (c) Sample inner circular arc from θ1 → θ0 at radius = r_conn_inner
            thetas_in = np.linspace(θ1, θ0, N + 1)
            inner_pts = [
                (r_conn_inner * np.cos(np.deg2rad(th)),
                 r_conn_inner * np.sin(np.deg2rad(th)))
                for th in thetas_in
            ]

            # (d) Startcap at θ0: semicircle radius = w_ann/2, center = (r_mid_ann, θ0)
            center_start = (
                r_mid_ann * np.cos(np.deg2rad(θ0)),
                r_mid_ann * np.sin(np.deg2rad(θ0))
            )
            phis_start = np.linspace(np.pi, 2 * np.pi, N + 1)
            cap_start_pts = [
                (
                    center_start[0] + (w_ann / 2) * np.cos(np.deg2rad(θ0) + φ),
                    center_start[1] + (w_ann / 2) * np.sin(np.deg2rad(θ0) + φ)
                )
                for φ in phis_start
            ]

            coords_conn = outer_pts + cap_end_pts + inner_pts + cap_start_pts
            conn_pad    = draw.Polygon(coords_conn)

        # -------------------------------------------------------------------------
        # Rotate + Translate this pad from RAW → FINAL (orientation + pos)
        # -------------------------------------------------------------------------
        conn_pad = draw.rotate([conn_pad], p.orientation, origin=(0, 0))[0]
        conn_pad = draw.translate([conn_pad], xoff=p.pos_x, yoff=p.pos_y)[0]

        # Add the connection‐pad metal
        self.add_qgeometry(
            'poly',
            { f'{name}_arc': conn_pad },
            layer=p.layer,
            subtract=False
        )

        #   This wedge has inner radius = r_conn_inner - pad_g, outer radius = r_conn_outer + pad_g.
        if np.isclose(cov_conn, 360.0):
            circ_big_gap   = draw.Point(0,0).buffer(r_gap_outer)
            circ_hole_gap  = draw.Point(0,0).buffer(r_gap_inner)
            gap_polygon    = draw.subtract(circ_big_gap, circ_hole_gap)
        else:
            Ng            = 120  # sampling resolution
            # (a) Outer arc of gap region at r = r_gap_outer, from θ0 → θ1
            thetas_out_g  = np.linspace(θ0, θ1, Ng + 1)
            outer_pts_g   = [
                (r_gap_outer * np.cos(np.deg2rad(th)),
                 r_gap_outer * np.sin(np.deg2rad(th)))
                for th in thetas_out_g
            ]

            # (b) Endcap of gap wedge at θ1 (semicircle radius = (r_gap_outer - r_gap_inner)/2)
            w_ann_gap     = r_gap_outer - r_gap_inner
            r_mid_gap     = 0.5 * (r_gap_outer + r_gap_inner)
            center_end_g  = (
                r_mid_gap * np.cos(np.deg2rad(θ1)),
                r_mid_gap * np.sin(np.deg2rad(θ1))
            )
            phis_end_g    = np.linspace(0, np.pi, Ng + 1)
            cap_end_pts_g = [
                (
                    center_end_g[0] + (w_ann_gap / 2) * np.cos(np.deg2rad(θ1) + φ),
                    center_end_g[1] + (w_ann_gap / 2) * np.sin(np.deg2rad(θ1) + φ)
                )
                for φ in phis_end_g
            ]

            # (c) Inner arc at radius = r_gap_inner, from θ1 → θ0
            thetas_in_g   = np.linspace(θ1, θ0, Ng + 1)
            inner_pts_g   = [
                (r_gap_inner * np.cos(np.deg2rad(th)),
                 r_gap_inner * np.sin(np.deg2rad(th)))
                for th in thetas_in_g
            ]

            # (d) Startcap at θ0 for gap
            center_start_g = (
                r_mid_gap * np.cos(np.deg2rad(θ0)),
                r_mid_gap * np.sin(np.deg2rad(θ0))
            )
            phis_start_g   = np.linspace(np.pi, 2*np.pi, Ng + 1)
            cap_start_pts_g = [
                (
                    center_start_g[0] + (w_ann_gap / 2) * np.cos(np.deg2rad(θ0) + φ),
                    center_start_g[1] + (w_ann_gap / 2) * np.sin(np.deg2rad(θ0) + φ)
                )
                for φ in phis_start_g
            ]

            coords_gap = outer_pts_g + cap_end_pts_g + inner_pts_g + cap_start_pts_g
            gap_polygon = draw.Polygon(coords_gap)

        # Rotate + translate the “gap” wedge exactly like the metal wedge
        gap_polygon = draw.rotate([gap_polygon], p.orientation, origin=(0, 0))[0]
        gap_polygon = draw.translate([gap_polygon], xoff=p.pos_x, yoff=p.pos_y)[0]

        # Add the gap with subtract=True
        self.add_qgeometry(
            'poly',
            { f'{name}_pad_gap': gap_polygon },
            layer=p.layer,
            subtract=True
        )

        # ─────────────── Place a QPin across the center angle, from r_conn_inner → r_conn_outer ───────────────
        θc_rad   = np.deg2rad(θ_center)
        # Raw (pre‐rotation) endpoints:
        raw_pin_in  = (r_conn_inner * np.cos(θc_rad), r_conn_inner * np.sin(θc_rad))
        raw_pin_out = (r_conn_outer * np.cos(θc_rad), r_conn_outer * np.sin(θc_rad))

        # Apply the same global rotation + translation to each endpoint:
        def _rotate_translate(pt):
            x0, y0 = pt
            θr = p.orientation * pi / 180.0
            xr = x0 * cos(θr) - y0 * sin(θr)
            yr = x0 * sin(θr) + y0 * cos(θr)
            return (xr + p.pos_x, yr + p.pos_y)

        

        # ─────────────── Build the CPW feedline in RAW coords ───────────────
        # CPW starts at radius = r_conn_outer along θ_center, and goes outward to r_conn_outer + cpw_len
        r_cpw_start = r_conn_outer
        r_cpw_end   = r_conn_outer + cpw_len

        raw_cpw_start = (r_cpw_start * np.cos(θc_rad), r_cpw_start * np.sin(θc_rad))
        raw_cpw_end   = (r_cpw_end   * np.cos(θc_rad), r_cpw_end   * np.sin(θc_rad))

        cpw_line = draw.LineString([ raw_cpw_start, raw_cpw_end ])

        # Rotate + translate that CPW line: RAW → FINAL
        cpw_line = draw.rotate([cpw_line], p.orientation, origin=(0, 0))[0]
        cpw_line = draw.translate([cpw_line], xoff=p.pos_x, yoff=p.pos_y)[0]


        pin_in  = _rotate_translate(raw_cpw_start)
        pin_out = _rotate_translate(raw_cpw_end)

        self.add_pin(
            name,                    # pin name = pad name
            points=np.array([pin_in, pin_out]),
            width=0.01,              # 0.01 μm (very thin) for port definition
            input_as_norm=True
        )

        # Add CPW center‐trace (path width = cpw_w)
        self.add_qgeometry(
            'path',
            { f'{name}_cpw': cpw_line },
            width=cpw_w,
            layer=p.layer,
            subtract=False
        )
        # Add CPW gap (path width = cpw_w + 2·cpw_g, subtract=True)
        self.add_qgeometry(
            'path',
            { f'{name}_cpw_gap': cpw_line },
            width=(cpw_w + 2 * cpw_g),
            layer=p.layer,
            subtract=True
        )
