"""Write Lightroom Classic / Camera Raw XMP files from Lightroom-scale settings.

* sidecar  - ``DSC01234.xmp`` next to the RAW: Lightroom/Camera Raw opens the
  photo with these settings applied (Metadata > Read Metadata from File).
* preset   - same content with a preset name; import in Lightroom via
  Develop > Presets > (+) > Import Presets.

Masks (people / sky / background), background blur and face retouch are
AI-only features of this editor and are not written (Lightroom would need its
own AI masks for them) - the returned note lists what was skipped.
"""
from __future__ import annotations

import uuid
from xml.sax.saxutils import quoteattr

from .lr_tools import parse_points
from .sliders import HSL_BANDS

_SIMPLE = {
    "tint": "Tint", "contrast": "Contrast2012", "highlights": "Highlights2012", "shadows": "Shadows2012",
    "whites": "Whites2012", "blacks": "Blacks2012", "texture": "Texture", "clarity": "Clarity2012",
    "dehaze": "Dehaze", "vibrance": "Vibrance", "saturation": "Saturation",
    "curve_highlights": "ParametricHighlights", "curve_lights": "ParametricLights",
    "curve_darks": "ParametricDarks", "curve_shadows": "ParametricShadows",
    "cg_shadows_hue": "SplitToningShadowHue", "cg_shadows_sat": "SplitToningShadowSaturation",
    "cg_shadows_lum": "ColorGradeShadowLum", "cg_midtones_hue": "ColorGradeMidtoneHue",
    "cg_midtones_sat": "ColorGradeMidtoneSat", "cg_midtones_lum": "ColorGradeMidtoneLum",
    "cg_highlights_hue": "SplitToningHighlightHue", "cg_highlights_sat": "SplitToningHighlightSaturation",
    "cg_highlights_lum": "ColorGradeHighlightLum", "cg_global_hue": "ColorGradeGlobalHue",
    "cg_global_sat": "ColorGradeGlobalSat", "cg_global_lum": "ColorGradeGlobalLum",
    "cg_blending": "ColorGradeBlending", "cg_balance": "SplitToningBalance",
    "sharpening": "Sharpness", "sharpness": "Sharpness", "sharpen_radius": "SharpenRadius",
    "sharpen_detail": "SharpenDetail", "sharpen_masking": "SharpenEdgeMasking",
    "nr_luminance": "LuminanceSmoothing", "nr_color": "ColorNoiseReduction",
    "defringe": "DefringePurpleAmount", "lens_distortion": "LensManualDistortionAmount",
    "lens_vignetting": "VignetteAmount", "vertical": "PerspectiveVertical",
    "horizontal": "PerspectiveHorizontal", "scale": "PerspectiveScale",
    "vignette": "PostCropVignetteAmount", "vignette_midpoint": "PostCropVignetteMidpoint",
    "vignette_feather": "PostCropVignetteFeather", "vignette_roundness": "PostCropVignetteRoundness",
    "vignette_highlights": "PostCropVignetteHighlightContrast",
    "grain": "GrainAmount", "grain_size": "GrainSize", "grain_roughness": "GrainFrequency",
    "shadow_tint": "ShadowTint", "red_hue": "RedHue", "red_sat": "RedSaturation",
    "green_hue": "GreenHue", "green_sat": "GreenSaturation", "blue_hue": "BlueHue", "blue_sat": "BlueSaturation",
}
for _b in HSL_BANDS:
    _cap = _b.capitalize()
    _SIMPLE[f"hue_{_b}"] = f"HueAdjustment{_cap}"
    _SIMPLE[f"sat_{_b}"] = f"SaturationAdjustment{_cap}"
    _SIMPLE[f"lum_{_b}"] = f"LuminanceAdjustment{_cap}"
    _SIMPLE[f"bw_{_b}"] = f"GrayMixer{_cap}"

# folded into other XMP values (Temperature/CropAngle/crop box) - not "skipped"
_FOLDED = ("warmth", "wb", "auto_level", "crop_aspect", "crop_zoom", "crop_focus", "auto_exposure")


def _fmt(attr: str, v) -> str:
    v = float(v)
    if attr in ("SharpenRadius",):
        return f"{v:+.1f}"
    if attr == "Exposure2012":
        return f"{v:+.2f}"
    return f"{int(round(v)):+d}" if attr not in ("PerspectiveScale",) else f"{int(round(v))}"


def _curve_seq(spec) -> str:
    pts = parse_points(spec) or [(0, 0), (1, 1)]
    items = "".join(f"<rdf:li>{int(round(x * 255))}, {int(round(y * 255))}</rdf:li>" for x, y in pts)
    return f"<rdf:Seq>{items}</rdf:Seq>"


def build_xmp(settings: dict, report: dict | None = None, is_raw: bool = True,
              preset_name: str | None = None) -> tuple[str, list[str]]:
    """Return (xml_text, skipped_setting_names)."""
    report = report or {}
    attrs = {"Version": "15.0", "ProcessVersion": "11.0", "HasSettings": "True"}
    if preset_name:
        attrs["PresetType"] = "Normal"
        attrs["UUID"] = uuid.uuid4().hex.upper()
        attrs["SupportsAmount"] = "False"
        attrs["SupportsColor"] = "True"
        attrs["SupportsMonochrome"] = "True"
        attrs["SupportsHighDynamicRange"] = "True"
        attrs["SupportsNormalDynamicRange"] = "True"
        attrs["SupportsSceneReferred"] = "True"
        attrs["SupportsOutputReferred"] = "True"
        attrs["CameraModelRestriction"] = ""
        attrs["Copyright"] = ""
        attrs["ContactInfo"] = ""
    skipped = []

    # white balance: the engine's final light estimate is the most faithful value
    wb = report.get("white_balance", {})
    temp = settings.get("temperature") or wb.get("neutralised_light_cct")
    if settings.get("wb") == "camera":
        attrs["WhiteBalance"] = "As Shot"
    elif temp and is_raw:
        attrs["WhiteBalance"] = "Custom"
        attrs["Temperature"] = str(int(round(float(temp))))
        attrs.setdefault("Tint", _fmt("Tint", settings.get("tint", 0)))
    elif settings.get("warmth"):
        attrs["IncrementalTemperature"] = _fmt("x", float(settings["warmth"]))
    # exposure: total EV actually applied (auto exposure + slider)
    ev = report.get("exposure_ev", settings.get("exposure"))
    if ev is not None:
        attrs["Exposure2012"] = _fmt("Exposure2012", ev)

    for k, v in settings.items():
        if v is None:
            continue
        if k in _SIMPLE:
            attrs[_SIMPLE[k]] = _fmt(_SIMPLE[k], v)
        elif k in ("remove_ca",):
            attrs["AutoLateralCA"] = "1" if str(v).lower() in ("1", "true", "on") else "0"
        elif k == "monochrome" or (k == "saturation" and float(v) <= -100) or \
                (k == "profile" and str(v).lower() == "monochrome"):
            attrs["ConvertToGrayscale"] = "True"
        elif k == "rotate":
            attrs["CropAngle"] = f"{-float(v):.2f}"
        elif k.startswith("point_curve") or k == "exposure" or k == "temperature" or k == "tint":
            continue
        elif k in _FOLDED:
            continue
        else:
            skipped.append(k)
    if settings.get("monochrome") or str(settings.get("profile", "")).lower() == "monochrome":
        attrs["ConvertToGrayscale"] = "True"
    if any(k.startswith("cg_") for k in settings):
        attrs.setdefault("ColorGradeBlending", "+50")
    # straighten detected automatically + crop rectangle from the render
    geo = report.get("geometry", {})
    if geo.get("rotate_deg") is not None:
        attrs["CropAngle"] = f"{-float(geo['rotate_deg']):.2f}"
    if "crop" in geo:
        t, l, b, r = geo["crop"]["normalised"]
        attrs.update({"HasCrop": "True", "CropTop": f"{t:.6f}", "CropLeft": f"{l:.6f}",
                      "CropBottom": f"{b:.6f}", "CropRight": f"{r:.6f}"})
    elif "CropAngle" in attrs:
        attrs["HasCrop"] = "True"

    curves = ""
    for k, tag in (("point_curve", "ToneCurvePV2012"), ("point_curve_red", "ToneCurvePV2012Red"),
                   ("point_curve_green", "ToneCurvePV2012Green"), ("point_curve_blue", "ToneCurvePV2012Blue")):
        if settings.get(k):
            curves += f"\n   <crs:{tag}>{_curve_seq(settings[k])}</crs:{tag}>"
    name = ""
    if preset_name:
        name = (f"\n   <crs:Name><rdf:Alt><rdf:li xml:lang=\"x-default\">{preset_name}</rdf:li></rdf:Alt></crs:Name>"
                f"\n   <crs:Group><rdf:Alt><rdf:li xml:lang=\"x-default\">Ai_photo_editor</rdf:li></rdf:Alt></crs:Group>")
    attr_txt = "\n    ".join(f"crs:{k}={quoteattr(str(v))}" for k, v in attrs.items())
    xml = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Ai_photo_editor">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    {attr_txt}>{name}{curves}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    return xml, sorted(set(skipped))


def write_xmp(path: str, settings: dict, report: dict | None = None, is_raw: bool = True,
              preset_name: str | None = None) -> dict:
    xml, skipped = build_xmp(settings, report, is_raw, preset_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return {"xmp": path, "not_in_lightroom_file": skipped}
