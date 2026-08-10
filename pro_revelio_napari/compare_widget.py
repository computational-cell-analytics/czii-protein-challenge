"""Dock widget that compares two maps, imitating ChimeraX's map-comparison workflow."""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from . import compare
from .widget import image_layers

MODE_SURFACE = "Surface coloured by difference"
MODE_DIFFERENCE = "Difference volume"
MODE_AGREEMENT = "Agreement overlay"
MODES = [MODE_SURFACE, MODE_DIFFERENCE, MODE_AGREEMENT]

SCALE_LABELS = {
    "match RMS (ChimeraX minRMS)": "min_rms",
    "z-score each map": "zscore",
    "leave amplitudes alone": "none",
}

# Refuse to pull anything bigger than this into RAM for an FFT.
SIZE_LIMIT = 2 * 1024**3


def diverging_colormap():
    from napari.utils import Colormap

    return Colormap(
        colors=[[0.13, 0.40, 0.67, 1.0], [0.97, 0.97, 0.97, 1.0], [0.72, 0.11, 0.11, 1.0]],
        controls=[0.0, 0.5, 1.0],
        name="pro_revelio_difference",
        display_name="blue-white-red",
    )


def agreement_colormap():
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(
        color_dict={
            None: (0, 0, 0, 0),
            0: (0, 0, 0, 0),
            1: (0.75, 0.75, 0.75, 1.0),  # both
            2: (0.72, 0.11, 0.11, 1.0),  # only A
            3: (0.13, 0.40, 0.67, 1.0),  # only B
        }
    )


class MapCompareWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        self._layout = QVBoxLayout(body)
        self._layout.setSpacing(8)

        self._build_maps_section()
        self._build_preprocessing_section()
        self._build_output_section()
        self._build_report_section()
        self._layout.addStretch(1)

        self._viewer.layers.events.inserted.connect(self._refresh_layers)
        self._viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    #
    # construction
    #

    def _section(self, title):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        self._layout.addWidget(box)
        return layout

    def _build_maps_section(self):
        layout = self._section("Maps")
        form = QFormLayout()
        layout.addLayout(form)

        self._combo_a = QComboBox()
        self._combo_b = QComboBox()
        form.addRow("Map A", self._combo_a)
        form.addRow("Map B", self._combo_b)

        swap = QPushButton("Swap A and B")
        swap.clicked.connect(self._swap)
        layout.addWidget(swap)

    def _build_preprocessing_section(self):
        layout = self._section("Preprocessing")
        form = QFormLayout()
        layout.addLayout(form)

        self._scale_combo = QComboBox()
        self._scale_combo.addItems(SCALE_LABELS)
        self._scale_combo.setToolTip(
            "Two maps rarely share an amplitude scale, so a raw subtraction mostly shows that "
            "mismatch. Matching the RMS is what ChimeraX's minRMS option does."
        )
        form.addRow("Amplitudes", self._scale_combo)

        self._match_band = QCheckBox("Match band-limit")
        self._match_band.setChecked(True)
        self._match_band.setToolTip(
            "Low-pass the sharper map to the blunter one's resolution. Without it the difference "
            "is dominated by frequencies only one map contains."
        )
        layout.addWidget(self._match_band)

        self._align = QCheckBox("Align map B onto map A (shift only)")
        self._align.setToolTip("Rigid translation from the cross-correlation peak. No rotation search.")
        layout.addWidget(self._align)

    def _build_output_section(self):
        layout = self._section("Output")
        form = QFormLayout()
        layout.addLayout(form)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(MODES)
        form.addRow("Show", self._mode_combo)

        self._level = QLabeledDoubleSlider(Qt.Horizontal)
        self._level.setRange(0.5, 6.0)
        self._level.setValue(2.0)
        self._level.setToolTip("Isosurface / overlap threshold, in standard deviations of map A.")
        form.addRow("Level (σ)", self._level)

        row = QHBoxLayout()
        self._run_button = QPushButton("Compare")
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)
        layout.addLayout(row)

    def _build_report_section(self):
        layout = self._section("Result")
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setMinimumHeight(210)
        self._report.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = self._report.font()
        font.setFamily("monospace")
        self._report.setFont(font)
        layout.addWidget(self._report)

    #
    # layer bookkeeping
    #

    def _refresh_layers(self, event=None):
        names = [layer.name for layer in self._source_layers()]
        for combo, fallback in ((self._combo_a, 0), (self._combo_b, -1)):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if previous in names:
                combo.setCurrentText(previous)
            elif names:
                combo.setCurrentIndex(fallback % len(names))
            combo.blockSignals(False)
        self._run_button.setEnabled(len(names) >= 2)

    def _source_layers(self):
        """Image layers except the ones this widget produced, so results can't feed back in."""
        return [l for l in image_layers(self._viewer) if not l.metadata.get("pro_revelio_compare")]

    def _swap(self):
        a, b = self._combo_a.currentText(), self._combo_b.currentText()
        self._combo_a.setCurrentText(b)
        self._combo_b.setCurrentText(a)

    def configure(self, map_a=None, map_b=None, scale=None, match_band=None, align=None, level=None):
        """Set the controls from code — used by the command line entry point."""
        if map_a:
            self._combo_a.setCurrentText(map_a)
        if map_b:
            self._combo_b.setCurrentText(map_b)
        if scale:
            for label, value in SCALE_LABELS.items():
                if value == scale:
                    self._scale_combo.setCurrentText(label)
        if match_band is not None:
            self._match_band.setChecked(match_band)
        if align is not None:
            self._align.setChecked(align)
        if level is not None:
            self._level.setValue(level)

    #
    # the comparison
    #

    def run(self):
        try:
            layer_a, layer_b = self._selected_layers()
            raw_a = np.asarray(layer_a.data, dtype=np.float32)
            raw_b = np.asarray(layer_b.data, dtype=np.float32)
        except Exception as error:
            self._report.setPlainText(f"Cannot compare these layers:\n\n{error}")
            return

        scale_mode = SCALE_LABELS[self._scale_combo.currentText()]
        cc_raw = float(np.corrcoef(raw_a.ravel(), raw_b.ravel())[0, 1])
        prepared_a, prepared_b, info = compare.prepare(
            raw_a,
            raw_b,
            scale=scale_mode,
            match_band=self._match_band.isChecked(),
            align=self._align.isChecked(),
        )

        level_sigma = self._level.value()
        stats = compare.compare_stats(prepared_a, prepared_b, level_sigma)
        difference = prepared_a - prepared_b

        mode = self._mode_combo.currentText()
        if mode == MODE_SURFACE:
            self._add_surface(prepared_a, difference, layer_a, level_sigma)
        elif mode == MODE_DIFFERENCE:
            self._add_difference(difference, layer_a, layer_b)
        else:
            self._add_agreement(prepared_a, prepared_b, layer_a, level_sigma)

        self._report.setPlainText(
            self._format_report(layer_a, layer_b, raw_a, cc_raw, scale_mode, info, stats, level_sigma)
        )

    def _selected_layers(self):
        names = {layer.name: layer for layer in self._source_layers()}
        name_a, name_b = self._combo_a.currentText(), self._combo_b.currentText()
        if name_a == name_b:
            raise ValueError("pick two different layers")
        layer_a, layer_b = names[name_a], names[name_b]
        if layer_a.data.shape != layer_b.data.shape:
            raise ValueError(
                f"shapes differ: {layer_a.data.shape} vs {layer_b.data.shape}.\n"
                "Resample the maps onto a common grid first — this widget does not regrid."
            )
        if layer_a.data.nbytes > SIZE_LIMIT:
            raise ValueError(f"{layer_a.name} is too large to compare in memory ({layer_a.data.nbytes / 1e9:.1f} GB)")
        return layer_a, layer_b

    #
    # output layers
    #

    def _add_surface(self, data, difference, layer_a, level_sigma):
        level = float(data.mean() + level_sigma * data.std())
        if not data.min() < level < data.max():
            raise ValueError("level is outside the data range")
        vertices, faces, values = compare.surface_by_value(data, difference, level)
        limit = float(np.percentile(np.abs(values), 99)) or 1.0

        self._replace(
            f"match: {layer_a.name}",
            lambda name: self._viewer.add_surface(
                (vertices, faces, values),
                name=name,
                scale=layer_a.scale,
                colormap=diverging_colormap(),
                contrast_limits=(-limit, limit),
                metadata={"pro_revelio_compare": True},
            ),
        )
        self._viewer.dims.ndisplay = 3

    def _add_difference(self, difference, layer_a, layer_b):
        limit = float(np.percentile(np.abs(difference), 99.5)) or 1.0
        self._replace(
            f"diff: {layer_a.name} − {layer_b.name}",
            lambda name: self._viewer.add_image(
                difference,
                name=name,
                scale=layer_a.scale,
                colormap=diverging_colormap(),
                contrast_limits=(-limit, limit),
                rendering="attenuated_mip",
                metadata={"pro_revelio_compare": True},
            ),
        )

    def _add_agreement(self, prepared_a, prepared_b, layer_a, level_sigma):
        labels = compare.agreement_labels(prepared_a, prepared_b, level_sigma)
        self._replace(
            f"agreement: {layer_a.name}",
            lambda name: self._viewer.add_labels(
                labels,
                name=name,
                scale=layer_a.scale,
                colormap=agreement_colormap(),
                metadata={"pro_revelio_compare": True},
            ),
        )

    def _replace(self, name, build):
        """Re-running should update the result layer, not stack another one on top."""
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
        return build(name)

    #
    # reporting
    #

    def _format_report(self, layer_a, layer_b, raw_a, cc_raw, scale_mode, info, stats, level_sigma):
        size = raw_a.shape[0]
        apix_a, apix_b = float(layer_a.scale[-1]), float(layer_b.scale[-1])
        lines = [
            f"A  {layer_a.name}",
            f"   {'x'.join(str(s) for s in layer_a.data.shape)} @ {apix_a:g} A/px",
            f"B  {layer_b.name}",
            f"   {'x'.join(str(s) for s in layer_b.data.shape)} @ {apix_b:g} A/px",
            "",
        ]

        if info["cutoffs"]:
            first, second = info["cutoffs"]
            lines.append(
                f"band-limit   A {compare.resolution(first, size, apix_a):.1f} A"
                f" | B {compare.resolution(second, size, apix_b):.1f} A"
            )
            if first != second:
                lines.append(f"             both filtered to {compare.resolution(info['cutoff'], size, apix_a):.1f} A")
        else:
            lines.append("band-limit   not matched")

        lines.append(f"amplitudes   {scale_mode}" + (f", B x {info['factor']:.3f}" if scale_mode != "none" else ""))
        shift = info["shift"]
        lines.append(
            "alignment    " + (f"shifted B by (z,y,x) = ({shift[0]:+.2f}, {shift[1]:+.2f}, {shift[2]:+.2f}) px"
                               if shift else "off")
        )
        lines += [
            "",
            f"CC raw               {cc_raw:.4f}",
            f"CC after preparation {stats['cc']:.4f}",
            f"RMS difference       {stats['rms']:.4f}",
            f"Dice at {level_sigma:.1f} sigma      {stats['dice']:.4f}",
            f"                     {stats['voxels_a']} vs {stats['voxels_b']} voxels above threshold",
        ]

        if abs(apix_a - apix_b) > 0.01 * max(apix_a, apix_b):
            lines += ["", f"WARNING pixel sizes differ by {abs(apix_a - apix_b) / max(apix_a, apix_b):.1%}"]
        return "\n".join(lines)
