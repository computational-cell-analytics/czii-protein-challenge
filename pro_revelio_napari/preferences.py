"""Register this plugin as napari's preferred reader for our formats.

Without this, napari's builtins also claim .h5/.tif/.zarr and every drag-and-drop pops a
"choose a reader" dialog.
"""

from .reader import ALL_EXTENSIONS, ZARR_EXTENSIONS

PLUGIN_NAME = "pro-revelio"


def reader_patterns():
    patterns = [f"*{ext}" for ext in ALL_EXTENSIONS]
    # napari appends a separator to directory paths before matching, so zarr needs its own pattern.
    patterns += [f"*{ext}/" for ext in ZARR_EXTENSIONS]
    return patterns


def is_default_reader():
    settings = _settings()
    return all(settings.get(p) == PLUGIN_NAME for p in reader_patterns())


def set_default_reader(enabled, override=False):
    """Claim (or release) the file patterns. Never steals a pattern assigned to another plugin."""
    from napari.settings import get_settings

    settings = get_settings()
    mapping = dict(settings.plugins.extension2reader)

    for pattern in reader_patterns():
        current = mapping.get(pattern)
        if enabled:
            if current is None or override or current == PLUGIN_NAME:
                mapping[pattern] = PLUGIN_NAME
        elif current == PLUGIN_NAME:
            mapping.pop(pattern)

    settings.plugins.extension2reader = mapping
    settings.save()


def _settings():
    from napari.settings import get_settings

    return dict(get_settings().plugins.extension2reader)
