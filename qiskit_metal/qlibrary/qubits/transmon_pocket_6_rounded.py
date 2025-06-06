import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import BaseQubit

class TransmonPocket6Rounded(BaseQubit):
    """Transmon pocket with 6 connection pads, now supporting rounded corners."""

    # ── Default drawing options, now with corner‐radius entries:
    default_options = Dict(
        pad_gap='30um',
        inductor_width='20um',
        pad_width='455um',
        pad_height='90um',
        pocket_width='650um',
        pocket_height='650um',

        # ── NEW: corner‐radius parameters (strings with units)
        pocket_radius='20um',
        pad_radius='15um',

        _default_connection_pads=Dict(
            pad_gap='15um',
            pad_width='125um',
            pad_height='30um',
            pad_cpw_shift='0um',
            pad_cpw_extent='25um',
            cpw_width='10um',
            cpw_gap='6um',
            cpw_extend='100um',
            pocket_extent='5um',
            pocket_rise='0um',
            loc_W='+1',
            loc_H='+1',
            pad_radius= '0um',
        )
    )

    component_metadata = Dict(
        short_name='Pocket',
        _qgeometry_table_path='True',
        _qgeometry_table_poly='True',
        _qgeometry_table_junction='True'
    )
    TOOLTIP = "Transmon pocket with 6 connection pads."

    def make(self):
        self.make_pocket()
        self.make_connection_pads()

    def make_pocket(self):
        """Makes standard transmon in a pocket, with rounded corners."""
        p = self.p

        # Parse sizes:
        pad_width   = p.pad_width
        pad_height  = p.pad_height
        pad_gap     = p.pad_gap
        pocket_w    = p.pocket_width
        pocket_h    = p.pocket_height
        pad_radius  = p.pad_radius
        pocket_radius = p.pocket_radius
        # Parse radii and clamp:
        R_pad    = min(pad_radius,
                       pad_width/2.0,
                       pad_height/2.0)
        R_pocket = min(pocket_radius,
                       pocket_w/2.0,
                       pocket_h/2.0)

        if (pad_width <= 2*R_pad) or (pad_height <= 2*R_pad):
            print(f"pad_radius={R_pad} is too large for pad {pad_width}×{pad_height}. Using shortest size ({R_pad}) for rounding.")

        if (pocket_w <= 2*R_pocket) or (pocket_h <= 2*R_pocket):
            print(f"pocket_radius={R_pocket} is too large for pocket {pocket_w}×{pocket_h}. Using shortest size ({R_pocket}) for rounding.")

        # ── Build rounded‐corner island pads:
        inner_pad = draw.rectangle(pad_width - 2*R_pad, pad_height - 2*R_pad)    # centered at (0,0)
        rounded_pad = inner_pad.buffer(R_pad, join_style=1)                      # full pad w×h, round corners

        pad_top = draw.translate(rounded_pad, 0, +((pad_height + pad_gap)/2.0))
        pad_bot = draw.translate(rounded_pad, 0, -((pad_height + pad_gap)/2.0))

        # Josephson junction line:
        rect_jj = draw.LineString([(0, -pad_gap/2.0), (0, +pad_gap/2.0)])

        # ── Build rounded‐corner pocket cutout:
        inner_pocket = draw.rectangle(pocket_w - 2*R_pocket, pocket_h - 2*R_pocket)
        rounded_pocket = inner_pocket.buffer(R_pocket, join_style=1)

        # Rotate + translate everything:
        polys = [rect_jj, pad_top, pad_bot, rounded_pocket]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)
        rect_jj, pad_top, pad_bot, rounded_pocket = polys

        # Add qgeometry:
        self.add_qgeometry('poly', dict(pad_top=pad_top, pad_bot=pad_bot))
        self.add_qgeometry('poly', dict(rect_pk=rounded_pocket), subtract=True)
        self.add_qgeometry('junction', dict(rect_jj=rect_jj),
                           width=p.inductor_width)

    def make_connection_pads(self):
        """Iterate through connector names and call make_connection_pad."""
        for name in self.options.connection_pads:
            self.make_connection_pad(name)

    def make_connection_pad(self, name: str):
        """Makes an individual connector, now with rounded corners."""
        p   = self.p
        pc  = self.p.connection_pads[name]

        cpw_width   = pc.cpw_width
        cpw_extend  = pc.cpw_extend
        pad_w_conn  = pc.pad_width
        pad_h_conn  = pc.pad_height
        pad_gap_conn= pc.pad_gap
        pocket_rise = pc.pocket_rise
        pocket_extent= pc.pocket_extent

        loc_W = float(pc.loc_W)
        loc_H = float(pc.loc_H)
        if loc_W not in (-1.0, 0.0, 1.0) or loc_H not in (-1.0, 0.0, 1.0):
            self.logger.info(
                f"Warning: loc_W={loc_W}, loc_H={loc_H} should be ±1 or 0."
            )

        # ── Determine radius for this connector pad:
        R_conn = float(pc.pad_radius)


        R_conn = min(R_conn, pad_w_conn/2.0, pad_h_conn/2.0)
        if (pad_w_conn <= 2*R_conn) or (pad_h_conn <= 2*R_conn):
            print(
                f"connection_pad_radius={R_conn} is too large for pad {pad_w_conn}×{pad_h_conn}.  Using shortest size ({R_conn}) for rounding."
            )

        # Build a centered, rounded‐corner pad of outer size pad_w_conn×pad_h_conn:
        inner_conn_w = pad_w_conn - 2*R_conn
        inner_conn_h = pad_h_conn - 2*R_conn
        inner_conn_rect = draw.rectangle(inner_conn_w, inner_conn_h)
        rounded_conn_pad = inner_conn_rect.buffer(R_conn, join_style=1)

        # Translate so that its lower‐left corner would match the old code’s placement:
        # Old straight code did: draw.rectangle(pad_w_conn, pad_h_conn, -pad_w_conn/2, pad_h_conn/2),
        # whose center is at (0, pad_h_conn).  So we shift by (0, pad_h_conn).
        connector_pad = draw.translate(rounded_conn_pad, 0, pad_h_conn)

        # ── Build the CPW “wire path” exactly as before:
        if loc_W != 0:
            connector_wire_path = draw.wkt.loads(f"""LINESTRING (\
                0 {pad_gap_conn + cpw_width/2.0}, \
                {pc.pad_cpw_extent}                           {pad_gap_conn + cpw_width/2.0}, \
                {(p.pocket_width - p.pad_width)/2 - pocket_extent} {pad_gap_conn + cpw_width/2.0 + pocket_rise}, \
                {(p.pocket_width - p.pad_width)/2 + cpw_extend}    {pad_gap_conn + cpw_width/2.0 + pocket_rise}\
            )""")
        else:
            connector_wire_path = draw.LineString([
                [0, pad_h_conn],
                [0, (p.pocket_width/2 - p.pad_height - p.pad_gap/2 - pad_gap_conn) + cpw_extend]
            ])

        # ── Flip and translate into the correct quadrant:
        objects = [connector_pad, connector_wire_path]
        if loc_W == 0:
            loc_Woff = 1.0
        else:
            loc_Woff = loc_W
        objects = draw.scale(objects, loc_Woff, loc_H, origin=(0, 0))
        objects = draw.translate(
            objects,
            loc_W * (p.pad_width)/2.0,
            loc_H * (p.pad_height + p.pad_gap/2.0 + pad_gap_conn)
        )
        objects = draw.rotate_position(objects, p.orientation, [p.pos_x, p.pos_y])
        connector_pad, connector_wire_path = objects

        # ── Add to qgeometry:
        self.add_qgeometry('poly', {f'{name}_connector_pad': connector_pad})
        self.add_qgeometry('path', {f'{name}_wire': connector_wire_path}, width=cpw_width)
        self.add_qgeometry('path', {f'{name}_wire_sub': connector_wire_path},
                           width=cpw_width + 2.0 * pc.cpw_gap, subtract=True)

        # ── Define the pin exactly as before:
        points = np.array(connector_wire_path.coords)
        self.add_pin(name,
                     points=points[-2:],
                     width=cpw_width,
                     input_as_norm=True)
