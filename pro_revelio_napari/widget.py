"""Dock widget with the 3D rendering controls for volumes opened through this plugin."""

import os

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from .preferences import is_default_reader, set_default_reader
from .reader import _is_supported

RENDERING_MODES = ["mip", "attenuated_mip", "minip", "average", "translucent", "additive", "iso"]
COLORMAPS = ["gray", "gray_r", "viridis", "magma", "inferno", "plasma", "turbo", "red", "green", "blue"]
ALL_LAYERS = "— all image layers —"

# Cap the number of voxels used for percentile estimates so auto-contrast stays interactive.
CONTRAST_SAMPLE = 4_000_000


class _DropArea(QFrame):
    """Accepts files dropped onto the widget itself, not just onto the canvas."""

    def __init__(self, on_paths):
        super().__init__()
        self._on_paths = on_paths
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(70)
        self.setStyleSheet("QFrame { border: 2px dashed palette(mid); border-radius: 6px; }")

        layout = QVBoxLayout(self)
        label = QLabel("Drop .h5 / .mrc / .mrcs / .tif / .zarr files here")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: none;")
        layout.addWidget(label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        paths = [p for p in paths if p]
        if paths:
            self._on_paths(paths)
            event.acceptProposedAction()


class VolumeViewerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._syncing = False

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

        self._build_open_section()
        self._build_display_section()
        self._build_target_section()
        self._build_rendering_section()
        self._build_contrast_section()
        self._build_appearance_section()
        self._build_camera_section()
        self._layout.addStretch(1)

        self._viewer.layers.events.inserted.connect(self._refresh_layers)
        self._viewer.layers.events.removed.connect(self._refresh_layers)
        self._viewer.layers.selection.events.active.connect(self._on_viewer_selection)
        self._refresh_layers()

    #
    # construction
    #

    def _section(self, title):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        self._layout.addWidget(box)
        return layout

    def _build_open_section(self):
        layout = self._section("Open")
        layout.addWidget(_DropArea(self._open_paths))

        buttons = QHBoxLayout()
        files_button = QPushButton("Open files…")
        files_button.clicked.connect(self._choose_files)
        folder_button = QPushButton("Open folder…")
        folder_button.clicked.connect(self._choose_folder)
        buttons.addWidget(files_button)
        buttons.addWidget(folder_button)
        layout.addLayout(buttons)

        # Claim the file patterns on first use, otherwise every drop asks which reader to use.
        if not is_default_reader():
            set_default_reader(True)
        self._default_reader_box = QCheckBox("Use this plugin for drag && drop")
        self._default_reader_box.setToolTip(
            "Registers Pro-Revelio as napari's preferred reader for these formats, so dropping a "
            "file does not ask which plugin to use."
        )
        self._default_reader_box.setChecked(is_default_reader())
        self._default_reader_box.toggled.connect(lambda on: set_default_reader(on, override=on))
        layout.addWidget(self._default_reader_box)

    def _build_display_section(self):
        layout = self._section("Display")

        row = QHBoxLayout()
        self._3d_button = QPushButton("3D view")
        self._3d_button.setCheckable(True)
        self._3d_button.setChecked(self._viewer.dims.ndisplay == 3)
        self._3d_button.toggled.connect(self._set_3d)
        row.addWidget(self._3d_button)

        self._grid_box = QCheckBox("Grid")
        self._grid_box.setToolTip("Tile the layers side by side instead of overlaying them.")
        self._grid_box.toggled.connect(lambda on: setattr(self._viewer.grid, "enabled", on))
        row.addWidget(self._grid_box)
        layout.addLayout(row)

        self._viewer.dims.events.ndisplay.connect(
            lambda e: self._3d_button.setChecked(self._viewer.dims.ndisplay == 3)
        )

    def _build_target_section(self):
        layout = self._section("Layer")
        self._layer_combo = QComboBox()
        self._layer_combo.currentIndexChanged.connect(self._on_target_changed)
        layout.addWidget(self._layer_combo)

    def _build_rendering_section(self):
        layout = self._section("Volume rendering")
        form = QFormLayout()
        layout.addLayout(form)

        self._rendering_combo = QComboBox()
        self._rendering_combo.addItems(RENDERING_MODES)
        # textActivated, not currentTextChanged: it also fires when the user re-picks the value
        # already shown, which is what "apply to all layers" needs.
        self._rendering_combo.textActivated.connect(
            lambda mode: self._apply(lambda layer: setattr(layer, "rendering", mode))
        )
        form.addRow("Mode", self._rendering_combo)

        self._attenuation = QLabeledDoubleSlider(Qt.Horizontal)
        self._attenuation.setRange(0.0, 2.0)
        self._attenuation.setValue(0.05)
        self._attenuation.valueChanged.connect(
            lambda v: self._apply(lambda layer: setattr(layer, "attenuation", v))
        )
        form.addRow("Attenuation", self._attenuation)

        self._iso_threshold = QLabeledDoubleSlider(Qt.Horizontal)
        self._iso_threshold.setRange(0.0, 1.0)
        self._iso_threshold.valueChanged.connect(
            lambda v: self._apply(lambda layer: setattr(layer, "iso_threshold", v))
        )
        form.addRow("Iso threshold", self._iso_threshold)

        self._rendering_combo.currentTextChanged.connect(self._update_rendering_enabled)

    def _build_contrast_section(self):
        layout = self._section("Contrast")

        self._contrast = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self._contrast.valueChanged.connect(self._on_contrast_changed)
        layout.addWidget(self._contrast)

        row = QHBoxLayout()
        self._low_pct = QSpinBox()
        self._low_pct.setRange(0, 49)
        self._low_pct.setValue(1)
        self._low_pct.setSuffix(" %")
        self._high_pct = QSpinBox()
        self._high_pct.setRange(51, 100)
        self._high_pct.setValue(99)
        self._high_pct.setSuffix(" %")
        auto_button = QPushButton("Auto contrast")
        auto_button.clicked.connect(self._auto_contrast)
        row.addWidget(QLabel("percentiles"))
        row.addWidget(self._low_pct)
        row.addWidget(self._high_pct)
        row.addWidget(auto_button)
        layout.addLayout(row)

    def _build_appearance_section(self):
        layout = self._section("Appearance")
        form = QFormLayout()
        layout.addLayout(form)

        self._colormap_combo = QComboBox()
        self._colormap_combo.addItems(COLORMAPS)
        self._colormap_combo.textActivated.connect(
            lambda name: self._apply(lambda layer: setattr(layer, "colormap", name))
        )
        form.addRow("Colormap", self._colormap_combo)

        self._gamma = QLabeledDoubleSlider(Qt.Horizontal)
        self._gamma.setRange(0.1, 2.0)
        self._gamma.setValue(1.0)
        self._gamma.valueChanged.connect(lambda v: self._apply(lambda layer: setattr(layer, "gamma", v)))
        form.addRow("Gamma", self._gamma)

        self._opacity = QLabeledDoubleSlider(Qt.Horizontal)
        self._opacity.setRange(0.0, 1.0)
        self._opacity.setValue(1.0)
        self._opacity.valueChanged.connect(lambda v: self._apply(lambda layer: setattr(layer, "opacity", v)))
        form.addRow("Opacity", self._opacity)

        self._z_scale = QLabeledDoubleSlider(Qt.Horizontal)
        self._z_scale.setRange(0.1, 10.0)
        self._z_scale.setValue(1.0)
        self._z_scale.setToolTip(
            "Multiplies the z voxel size — useful for tomograms whose z sampling is off or "
            "too flat to interpret in 3D."
        )
        self._z_scale.valueChanged.connect(self._on_z_scale_changed)
        form.addRow("Z stretch (×)", self._z_scale)

    def _build_camera_section(self):
        layout = self._section("Camera")

        self._angle_sliders = []
        form = QFormLayout()
        layout.addLayout(form)
        for i, axis in enumerate("XYZ"):
            slider = QLabeledDoubleSlider(Qt.Horizontal)
            slider.setRange(-180.0, 180.0)
            slider.valueChanged.connect(lambda _v, idx=i: self._on_angle_changed(idx))
            form.addRow(f"Rotate {axis}", slider)
            self._angle_sliders.append(slider)

        reset_button = QPushButton("Reset view")
        reset_button.clicked.connect(self._reset_view)
        layout.addWidget(reset_button)

        self._viewer.camera.events.angles.connect(self._sync_angles)
        self._sync_angles()

    #
    # opening files
    #

    def _choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open volumes",
            "",
            "Volumes (*.h5 *.hdf5 *.hdf *.mrc *.mrcs *.rec *.map *.st *.ali *.tif *.tiff);;All files (*)",
        )
        if paths:
            self._open_paths(paths)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open zarr / directory")
        if path:
            self._open_paths([path])

    def _open_paths(self, paths):
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                plugin = "pro-revelio" if _is_supported(path) else None  # None: let napari decide
                self._viewer.open(path, plugin=plugin)
            except Exception as error:
                self._viewer.status = f"Could not open {os.path.basename(path)}: {error}"

        if self._viewer.layers and self._viewer.dims.ndisplay == 2:
            self._3d_button.setChecked(True)

    #
    # target selection / syncing
    #

    def _image_layers(self):
        return image_layers(self._viewer)

    def _targets(self):
        name = self._layer_combo.currentText()
        if name == ALL_LAYERS:
            return self._image_layers()
        return [layer for layer in self._image_layers() if layer.name == name]

    def _apply(self, func):
        if self._syncing:
            return
        for layer in self._targets():
            try:
                func(layer)
            except Exception as error:
                self._viewer.status = f"{layer.name}: {error}"

    def _refresh_layers(self, event=None):
        names = [layer.name for layer in self._image_layers()]
        previous = self._layer_combo.currentText()

        self._syncing = True
        self._layer_combo.clear()
        self._layer_combo.addItems([ALL_LAYERS] + names)
        if previous in names:
            self._layer_combo.setCurrentText(previous)
        elif names:
            self._layer_combo.setCurrentText(names[-1])
        self._syncing = False

        self._sync_from_layer()

    def _on_viewer_selection(self, event=None):
        active = self._viewer.layers.selection.active
        if self._syncing or active is None:
            return
        index = self._layer_combo.findText(active.name)
        if index >= 0:
            self._layer_combo.setCurrentIndex(index)

    def _on_target_changed(self, _index):
        if self._syncing:
            return
        targets = self._targets()
        if len(targets) == 1:
            self._syncing = True
            self._viewer.layers.selection = {targets[0]}
            self._syncing = False
        self._sync_from_layer()

    def _sync_from_layer(self):
        """Pull the widget state from the current target so the controls never lie."""
        targets = self._targets()
        enabled = bool(targets)
        for widget in (
            self._rendering_combo,
            self._contrast,
            self._colormap_combo,
            self._gamma,
            self._opacity,
            self._z_scale,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            return

        layer = targets[0]
        self._syncing = True
        try:
            self._rendering_combo.setCurrentText(str(layer.rendering))
            self._gamma.setValue(float(layer.gamma))
            self._opacity.setValue(float(layer.opacity))
            self._z_scale.setValue(float(layer.scale[-3]) / _base_z(layer) if layer.ndim >= 3 else 1.0)
            self._attenuation.setValue(float(layer.attenuation))

            low, high = (float(v) for v in layer.contrast_limits_range)
            if high <= low:
                high = low + 1.0
            self._contrast.setRange(low, high)
            self._contrast.setValue(tuple(float(v) for v in layer.contrast_limits))
            self._iso_threshold.setRange(low, high)
            self._iso_threshold.setValue(float(layer.iso_threshold))

            name = getattr(layer.colormap, "name", "gray")
            if self._colormap_combo.findText(name) < 0:
                self._colormap_combo.addItem(name)
            self._colormap_combo.setCurrentText(name)
        finally:
            self._syncing = False

        self._update_rendering_enabled(self._rendering_combo.currentText())

    def _update_rendering_enabled(self, mode):
        self._attenuation.setEnabled(mode == "attenuated_mip")
        self._iso_threshold.setEnabled(mode == "iso")

    #
    # control callbacks
    #

    def _set_3d(self, on):
        self._viewer.dims.ndisplay = 3 if on else 2

    def _on_contrast_changed(self, value):
        low, high = value
        if high <= low:
            return
        self._apply(lambda layer: setattr(layer, "contrast_limits", [low, high]))

    def _on_z_scale_changed(self, factor):
        def set_z(layer):
            if layer.ndim < 3:
                return
            scale = list(layer.scale)
            scale[-3] = _base_z(layer) * factor
            layer.scale = scale

        self._apply(set_z)

    def _auto_contrast(self):
        for layer in self._targets():
            sample = _subsample(layer.data)
            low, high = np.percentile(sample, [self._low_pct.value(), self._high_pct.value()])
            if high <= low:
                continue
            layer.contrast_limits_range = [
                min(float(low), float(layer.contrast_limits_range[0])),
                max(float(high), float(layer.contrast_limits_range[1])),
            ]
            layer.contrast_limits = [float(low), float(high)]
        self._sync_from_layer()

    def _on_angle_changed(self, _index):
        if self._syncing:
            return
        self._viewer.camera.angles = tuple(slider.value() for slider in self._angle_sliders)

    def _sync_angles(self, event=None):
        self._syncing = True
        for slider, angle in zip(self._angle_sliders, self._viewer.camera.angles):
            slider.setValue(float(angle))
        self._syncing = False

    def _reset_view(self):
        self._viewer.reset_view()
        self._sync_angles()


def image_layers(viewer):
    from napari.layers import Image

    return [layer for layer in viewer.layers if isinstance(layer, Image)]


def _base_z(layer):
    """The z voxel size the file was opened with — the Z stretch slider multiplies this."""
    base = layer.metadata.setdefault("pro_revelio_base_z", float(layer.scale[-3]))
    return base or 1.0


def _subsample(data):
    """Percentiles from a strided slab — enough for contrast, cheap for a lazy tomogram."""
    if data.ndim >= 3:
        # Size the slab by plane area, so a big tomogram reads a couple of slices, not a whole block.
        plane = int(np.prod(data.shape[1:]))
        count = int(np.clip(CONTRAST_SAMPLE // max(plane, 1), 1, min(16, data.shape[0])))
        start = max(0, data.shape[0] // 2 - count // 2)
        data = data[start: start + count]

    step = max(1, int(np.ceil((data.size / CONTRAST_SAMPLE) ** (1 / max(data.ndim, 1)))))
    sample = np.asarray(data[(slice(None, None, step),) * data.ndim])
    return sample[np.isfinite(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
