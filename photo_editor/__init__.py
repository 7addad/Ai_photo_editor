"""AI photo editor for RAW files (Sony ARW and every format LibRaw reads)."""
from .presets import PRESETS, list_presets, resolve
from .raw_io import load_image, save_image

__all__ = ["PRESETS", "list_presets", "resolve", "load_image", "save_image", "edit"]
__version__ = "0.3.0"


def edit(path: str, preset: str = "natural", output: str | None = None, sliders: dict | None = None,
         **overrides):
    """One-call API: load, edit with a preset (+ overrides), save. Returns the report.

    ``sliders`` are the regular-editing sliders (Lightroom-like scales), e.g.
    ``edit("a.ARW", "standard", sliders={"contrast": 20, "temperature": 5200, "sharpness": 60})``.
    """
    import os
    from .pipeline import process
    from .sliders import sliders_to_params
    data = load_image(path)
    res = process(data, resolve(preset, {**overrides, **sliders_to_params(sliders or {})}))
    output = output or os.path.splitext(path)[0] + f"_{preset}.jpg"
    save_image(res.srgb, output, data)
    res.report["output"] = output
    return res.report
