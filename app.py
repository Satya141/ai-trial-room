"""Gradio UI for AI Trial Room.

Run locally::

    python app.py

Or on Kaggle / Colab with a public link::

    AITR_SHARE=1 python app.py

Design notes
------------
The UI is deliberately opinionated about two things:

*Consent is blocking.* The Generate button stays disabled until the consent
checkbox is ticked. Uploading someone else's photo to a try-on tool is the most
likely way this app gets misused, so the gate is structural rather than a
notice in small print.

*Failure messages are actionable.* Every error path surfaces a
:class:`~ai_trial_room.utils.errors.TrialRoomError` user message - "use a
full-length photo", not a stack trace. In a shop, the person operating this is
a salesperson, not an engineer.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from PIL import Image

from ai_trial_room.backends.base import TryOnOptions
from ai_trial_room.backends.registry import loaded_backends, unload_all
from ai_trial_room.config import (
    APP_TAGLINE,
    APP_TITLE,
    CONFIG,
    CONSENT_TEXT,
    EXAMPLES_DIR,
    MODEL_SPECS,
    BackendId,
    Category,
    DrapeStyle,
    DupattaStyle,
    QualityPreset,
)
from ai_trial_room.lora import NO_LORA_LABEL, lora_choices
from ai_trial_room.preprocessing.parsing import provider_report
from ai_trial_room.router import describe_routing, run_try_on
from ai_trial_room.spaces_support import (
    apply_space_overrides,
    detect_space,
    estimate_duration,
    gpu,
    reset_hardware_cache,
    space_footer,
)
from ai_trial_room.utils.device import detect_hardware, vram_report
from ai_trial_room.utils.errors import InvalidInputError, TrialRoomError
from ai_trial_room.utils.fetch import fetch_image
from ai_trial_room.utils.image_io import (
    download_cache_dir,
    make_side_by_side,
    purge_download_cache,
)
from ai_trial_room.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

CATEGORY_CHOICES = [category.label for category in Category]
DRAPE_CHOICES = [style.label for style in DrapeStyle]
DUPATTA_CHOICES = [style.label for style in DupattaStyle]
PRESET_CHOICES = [preset.label for preset in QualityPreset]

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.rose,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CUSTOM_CSS = """
#aitr-header { text-align: center; padding: 0.5rem 0 1rem; }
#aitr-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700;
  background: linear-gradient(95deg, #be123c, #f59e0b);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#aitr-header p { margin: 0.3rem 0 0; opacity: 0.72; font-size: 1rem; }
.aitr-notice { font-size: 0.85rem; opacity: 0.75; text-align: center;
  padding-top: 0.4rem; }
#aitr-generate { font-size: 1.05rem; font-weight: 600; }
footer { display: none !important; }
"""


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def on_category_change(category_label: str) -> tuple[Any, Any, Any]:
    """Show the style controls relevant to the chosen category.

    Saree gets a drape-style dropdown, lehenga gets a dupatta-style dropdown, and
    both draped categories get a full-length-photo reminder.

    Parameters
    ----------
    category_label:
        Selected UI label.

    Returns
    -------
    tuple
        Visibility updates for ``(drape, dupatta, framing_hint)``.
    """
    try:
        category = Category.from_label(category_label)
    except ValueError:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    return (
        gr.update(visible=category is Category.SAREE),
        gr.update(visible=category is Category.LEHENGA),
        gr.update(visible=category.is_draped),
    )


def on_garment_url(url: str) -> tuple[Any, str]:
    """Fetch a garment image from a pasted link.

    Accepts a direct image URL or a shop's product page, from which the main
    product image is pulled out of the page's link-preview metadata.

    Parameters
    ----------
    url:
        Whatever the user pasted.

    Returns
    -------
    tuple
        ``(garment_image_update, status_markdown)``. On failure the image is left
        untouched so a bad paste does not clear a good upload.
    """
    if not (url or "").strip():
        return gr.update(), ""

    try:
        fetched = fetch_image(url)
    except TrialRoomError as exc:
        logger.info("Garment URL rejected: %s | %s", exc.user_message, exc.detail)
        return gr.update(), f"⚠️ {exc.user_message}"
    except Exception as exc:  # noqa: BLE001 - last resort for a user-supplied URL
        logger.exception("Unexpected failure fetching a garment URL")
        return gr.update(), f"⚠️ Could not load that link. ({type(exc).__name__})"

    note = " (from the product page)" if fetched.from_product_page else ""
    return (
        gr.update(value=fetched.image),
        f"✅ Garment loaded{note} - {fetched.image.width}×{fetched.image.height}.",
    )


def on_preset_change(preset_label: str) -> Any:
    """Sync the refine checkbox to the chosen preset.

    Refinement is what makes "Best" slower, so the checkbox should reflect the
    preset rather than silently disagreeing with it. The user can still override
    it afterwards.
    """
    try:
        preset = QualityPreset.from_label(preset_label)
    except ValueError:
        return gr.update()
    return gr.update(value=preset.settings().refine)


def on_consent_change(consented: bool) -> Any:
    """Enable the Generate button only once consent is given."""
    return gr.update(interactive=bool(consented))


def free_memory() -> str:
    """Unload all backends and report the VRAM state, for the Advanced panel."""
    unload_all()
    return f"GPU memory released. {vram_report()}"


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    """Read one of :func:`generate`'s arguments by name from ``*args``/``**kwargs``.

    The positional index is looked up from the live signature rather than
    hardcoded, so inserting a control into the UI cannot silently make the
    ZeroGPU duration estimate read the wrong slot.

    Returns
    -------
    Any
        The argument value, or ``None`` when it was not supplied.
    """
    if name in kwargs:
        return kwargs[name]
    index = _GENERATE_PARAM_INDEX.get(name)
    if index is None or index >= len(args):
        return None
    return args[index]


def _gpu_duration(*args: Any, **kwargs: Any) -> int:
    """Estimate this request's GPU seconds, for the ZeroGPU quota.

    ZeroGPU kills a call that outruns its declared duration, so the estimate is
    derived from the request's own settings rather than a fixed ceiling: a Best
    preset with refinement needs roughly four times a Fast one. Overestimating
    only costs queue priority, so every fallback errs high.
    """
    try:
        preset_label = _arg(args, kwargs, "preset_label") or CONFIG.default_preset.label
        raw_steps = _arg(args, kwargs, "steps")
        refine = bool(_arg(args, kwargs, "refine"))

        settings = QualityPreset.from_label(str(preset_label)).settings()
        steps = int(raw_steps) if raw_steps else settings.steps

        return estimate_duration(
            steps,
            CONFIG.default_backend.value,
            refine=refine or settings.refine,
        )
    except Exception:  # noqa: BLE001 - a bad estimate must not block the request
        return estimate_duration(40, CONFIG.default_backend.value, refine=True)


@gpu(duration=_gpu_duration)
def generate(
    person_image: Image.Image | None,
    garment_image: Image.Image | None,
    category_label: str,
    drape_label: str,
    dupatta_label: str,
    preset_label: str,
    backend_label: str,
    lora_label: str,
    lora_weight: float,
    steps: int,
    true_cfg: float,
    seed: int,
    preserve_face: bool,
    identity_strength: float,
    harmonize_colors: bool,
    refine: bool,
    sharpen: float,
    extra_prompt: str,
    consent: bool,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, Any, Any, str]]:
    """Run one try-on and yield UI updates.

    Yields rather than returns so the download file and status text can update
    together with the images.

    Parameters
    ----------
    person_image, garment_image:
        Uploads from the two image components.
    category_label, drape_label, dupatta_label, preset_label, backend_label:
        Dropdown and radio selections (UI labels, not enum values).
    lora_label, lora_weight:
        Selected drape adapter and its strength.
    steps, true_cfg, seed:
        Advanced sampler settings. ``steps`` of 0 means "use the preset".
    preserve_face, identity_strength, harmonize_colors:
        Identity and colour postprocessing controls.
    refine, sharpen:
        Region-refinement pass and garment sharpening amount.
    extra_prompt:
        Optional free-text instruction appended to the generated prompt.
    consent:
        State of the consent checkbox.
    progress:
        Gradio progress tracker, injected by Gradio.

    Yields
    ------
    tuple
        ``(result_image, comparison_image, download_file, status_markdown)``.
    """
    def report(fraction: float, message: str) -> None:
        """Bridge the backend progress callback onto Gradio's tracker."""
        progress(fraction, desc=message)

    # On ZeroGPU the GPU only exists inside this call, so the hardware snapshot
    # cached during startup says "no GPU" and would pick the wrong dtype.
    reset_hardware_cache()

    # Sweep any previous result that has outlived its download window.
    purge_download_cache()

    try:
        # Dropdown values are trusted in normal use, but the Gradio API surface
        # accepts arbitrary strings, so parse defensively and report cleanly.
        try:
            category = Category.from_label(category_label)
            drape_style = DrapeStyle.from_label(drape_label)
            dupatta_style = DupattaStyle.from_label(dupatta_label)
            preset = QualityPreset.from_label(preset_label)
        except ValueError as exc:
            raise InvalidInputError(
                "That category or style option is not recognised. Please pick one "
                "from the dropdowns.",
                detail=str(exc),
            ) from exc

        backend_id = _resolve_backend_label(backend_label)

        # The preset supplies the baseline; Advanced settings override it. A
        # steps value of 0 means "leave the preset alone".
        options = TryOnOptions.from_preset(
            preset,
            steps=int(steps) or None,
            true_cfg_scale=float(true_cfg) or None,
            seed=int(seed),
            drape_style=drape_style,
            dupatta_style=dupatta_style,
            extra_prompt=extra_prompt or "",
            preserve_face=bool(preserve_face),
            harmonize_colors=bool(harmonize_colors),
            refine=bool(refine),
            sharpen=float(sharpen),
            lora_name=None if lora_label == NO_LORA_LABEL else lora_label,
            lora_weight=float(lora_weight),
        )

        # Identity strength is global config rather than per-request, so apply it
        # here for this generation.
        CONFIG.identity.strength = float(identity_strength)

        report(0.01, "Starting...")
        report_obj = run_try_on(
            person_image,
            garment_image,
            category,
            options=options,
            backend_id=backend_id,
            consent=bool(consent),
            progress=report,
        )

        comparison = make_side_by_side(report_obj.before, report_obj.after)

        # Gradio hands the browser a path that is fetched after this handler
        # returns, so the file cannot live in an ephemeral_dir - it would be
        # gone before the user clicks Download. It goes in the swept download
        # cache instead: outside the project tree, deleted on process exit, and
        # purged by TTL at the top of this function.
        download_path = (
            download_cache_dir()
            / f"ai-trial-room-{category.value}-{report_obj.result.seed}.png"
        )
        report_obj.after.save(download_path, format="PNG", optimize=True)

        yield (
            report_obj.after,
            comparison,
            gr.update(value=str(download_path), visible=True),
            report_obj.caption(),
        )

    except TrialRoomError as exc:
        logger.warning("Try-on rejected: %s | detail=%s", exc.user_message, exc.detail)
        yield None, None, gr.update(visible=False), f"⚠️ {exc.user_message}"

    except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
        logger.exception("Unexpected failure during generation")
        yield (
            None,
            None,
            gr.update(visible=False),
            "⚠️ Something went wrong. Please try again, or check the logs if this "
            f"keeps happening. ({type(exc).__name__})",
        )


#: Positional index of each of :func:`generate`'s parameters, derived from its
#: signature so :func:`_arg` cannot drift when a UI control is added.
_GENERATE_PARAM_INDEX: dict[str, int] = {
    name: index
    for index, name in enumerate(inspect.signature(generate).parameters)
}


def _backend_choices() -> list[str]:
    """Build the backend dropdown labels, including the automatic option."""
    from ai_trial_room.backends.registry import available_backends

    labels = ["Automatic (recommended)"]
    for backend_id in available_backends():
        spec = MODEL_SPECS[backend_id]
        marker = "" if spec.is_commercial else "  [research only]"
        labels.append(f"{spec.repo_id}{marker}")
    return labels


def _resolve_backend_label(label: str) -> BackendId | None:
    """Map a backend dropdown label back to a :class:`BackendId`.

    Returns
    -------
    BackendId or None
        ``None`` for the automatic option, letting the router decide.
    """
    if not label or label.startswith("Automatic"):
        return None
    repo = label.split("  [")[0].strip()
    for backend_id, spec in MODEL_SPECS.items():
        if spec.repo_id == repo:
            return backend_id
    return None


def _example_rows() -> list[list[Any]]:
    """Collect example person/garment pairs from ``assets/examples``.

    Expected naming, so examples are self-describing on disk::

        assets/examples/person_01.jpg
        assets/examples/saree_01.jpg
        assets/examples/kurti_01.jpg

    Returns
    -------
    list
        Rows of ``[person_path, garment_path, category_label]``. Empty when no
        examples are present, in which case the Examples block is hidden.
    """
    if not EXAMPLES_DIR.exists():
        return []

    people = sorted(EXAMPLES_DIR.glob("person_*"))
    if not people:
        return []

    rows: list[list[Any]] = []
    for category in Category:
        garments = sorted(EXAMPLES_DIR.glob(f"{category.value}_*"))
        for index, garment in enumerate(garments):
            person = people[index % len(people)]
            rows.append([str(person), str(garment), category.label])
    return rows


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks app.

    Returns
    -------
    gr.Blocks
        The unlaunched interface, so tests and scripts can build it without
        starting a server.
    """
    hardware = detect_hardware()
    runtime = CONFIG.runtime

    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title=APP_TITLE) as demo:
        gr.HTML(
            f"""
            <div id="aitr-header">
              <h1>{APP_TITLE}</h1>
              <p>{APP_TAGLINE} &mdash; dresses, kurtis, kurtas, sarees &amp; lehengas</p>
            </div>
            """
        )

        with gr.Row():
            # ---------------- Inputs ---------------- #
            with gr.Column(scale=4):
                with gr.Row():
                    person_input = gr.Image(
                        label="1. Photo of the person",
                        type="pil",
                        height=320,
                        sources=["upload", "webcam", "clipboard"],
                    )
                    garment_input = gr.Image(
                        label="2. Garment photo",
                        type="pil",
                        height=320,
                        sources=["upload", "webcam", "clipboard"],
                    )

                with gr.Row():
                    garment_url_input = gr.Textbox(
                        label="…or paste a garment link",
                        placeholder="https://shop.example/products/kanjivaram-silk-saree",
                        scale=4,
                        max_lines=1,
                    )
                    fetch_button = gr.Button("Load link", scale=1)
                url_status = gr.Markdown("", elem_classes=["aitr-notice"])

                framing_hint = gr.Markdown(
                    "**Tip:** for sarees and lehengas use a full-length photo "
                    "(head to at least the knees), standing, facing the camera.",
                    elem_classes=["aitr-notice"],
                    visible=True,
                )

                with gr.Row():
                    category_input = gr.Dropdown(
                        CATEGORY_CHOICES,
                        value=Category.SAREE.label,
                        label="3. Category",
                        scale=1,
                    )
                    drape_input = gr.Dropdown(
                        DRAPE_CHOICES,
                        value=DrapeStyle.NIVI.label,
                        label="Drape style",
                        visible=True,
                        scale=1,
                    )
                    dupatta_input = gr.Dropdown(
                        DUPATTA_CHOICES,
                        value=DupattaStyle.SINGLE_SHOULDER.label,
                        label="Dupatta style",
                        visible=False,
                        scale=1,
                    )

                preset_input = gr.Radio(
                    PRESET_CHOICES,
                    value=CONFIG.default_preset.label,
                    label="4. Quality",
                    info="Best adds a second pass over the pallu / dupatta for "
                    "sharper border and zari detail.",
                )

                consent_input = gr.Checkbox(
                    label=CONSENT_TEXT,
                    value=False,
                )

                generate_button = gr.Button(
                    "Generate try-on",
                    variant="primary",
                    size="lg",
                    interactive=False,
                    elem_id="aitr-generate",
                )

                with gr.Accordion("Advanced settings", open=False):
                    backend_input = gr.Dropdown(
                        _backend_choices(),
                        value="Automatic (recommended)",
                        label="Model backend",
                        info="Automatic picks a commercially licensed model that "
                        "supports your chosen category.",
                    )
                    lora_choice_list = lora_choices()
                    lora_input = gr.Dropdown(
                        lora_choice_list,
                        value=NO_LORA_LABEL,
                        label="Drape LoRA",
                        info=(
                            "Trained adapters found in loras/. The trigger token is "
                            "added to the prompt automatically."
                            if len(lora_choice_list) > 1
                            else "No adapters found in loras/. Train one with "
                            "scripts/train_saree_lora.py (see datasets/README.md)."
                        ),
                        interactive=len(lora_choice_list) > 1,
                    )
                    lora_weight_input = gr.Slider(
                        0.0, 1.5, value=1.0, step=0.05,
                        label="LoRA strength",
                        info="Above ~1.2 the drape LoRA starts overriding the "
                        "garment reference.",
                    )
                    steps_input = gr.Slider(
                        0, 60, value=0, step=1,
                        label="Inference steps",
                        info="0 = use the Quality preset. Higher = finer fabric "
                        "detail, slower generation.",
                    )
                    cfg_input = gr.Slider(
                        0.0, 8.0, value=0.0, step=0.1,
                        label="Guidance scale (true CFG)",
                        info="0 = use the Quality preset. Above ~6 starts to look "
                        "over-processed.",
                    )
                    seed_input = gr.Number(
                        value=-1, precision=0, label="Seed",
                        info="-1 for a new random result each time. Reuse a seed "
                        "to reproduce an exact output.",
                    )
                    with gr.Row():
                        face_input = gr.Checkbox(
                            value=True, label="Preserve original face",
                        )
                        harmonize_input = gr.Checkbox(
                            value=True, label="Match colour & lighting",
                        )
                    identity_input = gr.Slider(
                        0.5, 1.0, value=CONFIG.identity.strength, step=0.01,
                        label="Identity strength",
                        info="How strongly the original face is preserved. Lower "
                        "lets more of the generated lighting through.",
                    )
                    with gr.Row():
                        refine_input = gr.Checkbox(
                            value=False,
                            label="Refine pallu / dupatta (slower)",
                        )
                        sharpen_input = gr.Slider(
                            0.0, 0.8, value=0.35, step=0.05,
                            label="Garment sharpening",
                        )
                    extra_input = gr.Textbox(
                        label="Extra instruction (optional)",
                        placeholder="e.g. add a gold potli bag, or keep the "
                        "dupatta off the shoulder",
                        lines=2,
                    )
                    free_button = gr.Button("Free GPU memory", size="sm")
                    memory_status = gr.Markdown("")

            # ---------------- Outputs ---------------- #
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Result"):
                        result_output = gr.Image(
                            label="Try-on result", type="pil", height=520,
                            show_download_button=True,
                        )
                    with gr.Tab("Before / after"):
                        comparison_output = gr.Image(
                            label="Original vs. try-on", type="pil", height=520,
                        )

                status_output = gr.Markdown("")
                download_output = gr.File(
                    label="Download result", visible=False, interactive=False,
                )
                gr.Markdown(
                    f"🔒 {CONFIG.privacy.notice}  \n"
                    f"🖥️ {hardware.describe()}",
                    elem_classes=["aitr-notice"],
                )

        # ---------------- Examples ---------------- #
        examples = _example_rows()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[person_input, garment_input, category_input],
                label="Example combinations (click to load)",
                examples_per_page=8,
            )
        else:
            gr.Markdown(
                f"_Add example images to `{EXAMPLES_DIR.relative_to(Path.cwd()) if EXAMPLES_DIR.is_relative_to(Path.cwd()) else EXAMPLES_DIR}` "
                "named `person_01.jpg`, `saree_01.jpg`, `kurti_01.jpg` … to "
                "populate a one-click examples gallery._",
                elem_classes=["aitr-notice"],
            )

        # ---------------- About ---------------- #
        with gr.Accordion("About, models & licensing", open=False):
            gr.Markdown(_about_markdown())
            gr.Dataframe(
                value=[list(row.values()) for row in describe_routing()],
                headers=list(describe_routing()[0].keys()),
                label="Category routing",
                interactive=False,
                wrap=True,
            )

        # ---------------- Wiring ---------------- #
        category_input.change(
            on_category_change,
            inputs=category_input,
            outputs=[drape_input, dupatta_input, framing_hint],
        )
        for trigger in (fetch_button.click, garment_url_input.submit):
            trigger(
                on_garment_url,
                inputs=garment_url_input,
                outputs=[garment_input, url_status],
            )
        preset_input.change(on_preset_change, preset_input, refine_input)
        consent_input.change(on_consent_change, consent_input, generate_button)
        free_button.click(free_memory, outputs=memory_status)

        generate_button.click(
            generate,
            inputs=[
                person_input, garment_input, category_input, drape_input,
                dupatta_input, preset_input, backend_input, lora_input,
                lora_weight_input, steps_input,
                cfg_input, seed_input, face_input, identity_input,
                harmonize_input, refine_input, sharpen_input, extra_input,
                consent_input,
            ],
            outputs=[result_output, comparison_output, download_output, status_output],
            concurrency_limit=1,  # one heavy generation at a time per GPU
        )

    return demo


def _about_markdown() -> str:
    """Render the About panel, including the live license table."""
    rows = "\n".join(
        f"| `{spec.repo_id}` | {spec.license_name} | "
        f"{'✅ commercial' if spec.is_commercial else '⚠️ research only'} | "
        f"~{spec.vram_gb_bf16:.0f} GB |"
        for spec in MODEL_SPECS.values()
    )
    parsing = provider_report()
    return f"""
### How it works

1. **Preprocess** - your photo is letterboxed (never cropped, so a saree hem
   survives), pose is detected with MediaPipe, and a human-parsing model marks
   which pixels are existing clothing.
2. **Route** - draped garments (saree, lehenga) always go to a reference-editing
   model, because garment-warping models cannot physically represent a drape.
3. **Generate** - the person and garment are passed as two reference images
   alongside a prompt that names the specific garment components (pleats, pallu,
   choli, dupatta) the style requires.
4. **Postprocess** - your original face is blended back in, and colour and
   lighting are matched to your photo.

### Models & licences

| Model | Licence | Commercial use | VRAM (bf16) |
|---|---|---|---|
{rows}

### Supporting models - all commercially usable

| Component | Model | Licence |
|---|---|---|
| Pose & face | MediaPipe Pose / FaceMesh | Apache-2.0 ✅ |
| Human parsing | {parsing["Provider"]} | {parsing["Licence"]} {"✅" if parsing["Commercial use"] == "yes" else "⚠️"} |
| Garment cut-out | rembg + u2net | MIT ✅ |

Human parsing is pluggable. The default is MediaPipe Selfie Multiclass
(Apache-2.0), whose coarse `clothes` class is split into upper and lower garments
using the detected hip line - which is what keeps a kurti mask off your trousers.
The finer-grained SegFormer model is research-licensed and disabled unless
`ALLOW_NONCOMMERCIAL=1`.

### Privacy

Uploads are held in memory and in a temporary directory that is deleted when
the request finishes. Results have all EXIF metadata stripped, so a downloaded
image cannot leak where the original photo was taken. Nothing is logged to disk
beyond timings.

### Known limitations

- Heavy occlusion (arms folded across the body) confuses the garment mask.
- Very fine zari and mirror work can soften; raise inference steps to help.
- Nauvari (dhoti-style) drapes are the least reliable - the training data for
  that silhouette is thin. Phase 3's LoRA fine-tune targets exactly this.
- Seated and three-quarter-turned poses are less reliable than standing front-on.
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Parse arguments, configure logging and launch the server."""
    parser = argparse.ArgumentParser(description=f"{APP_TITLE} - {APP_TAGLINE}")
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link (needed on Kaggle/Colab).",
    )
    parser.add_argument("--port", type=int, default=CONFIG.server_port)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-level", default=CONFIG.log_level)
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("%s v%s starting", APP_TITLE, __import__("ai_trial_room").__version__)

    # Must run before build_ui(), which reads the config it adjusts.
    apply_space_overrides()
    info = detect_space()
    logger.info("Hardware: %s", detect_hardware().describe())
    logger.info(
        "Commercial-only mode: %s | loaded backends: %s",
        not CONFIG.allow_noncommercial,
        loaded_backends() or "none",
    )

    demo = build_ui()

    # A Space supplies its own host and port; passing ours breaks the proxy.
    launch_kwargs: dict[str, Any] = {"show_api": False, "show_error": True}
    if not info.on_space:
        launch_kwargs.update(
            share=args.share or CONFIG.share_gradio,
            server_name=args.host,
            server_port=args.port,
        )

    demo.queue(max_size=12).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
