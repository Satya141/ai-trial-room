"""Prompt templates for the reference-editing backends.

This module is where the "specialised for Indian wear" claim is actually
earned. A generic instruction like *"put this garment on the person"* produces
a saree that looks like a bedsheet: no pleats, pallu on the wrong shoulder,
blouse missing. The templates here name the specific garment components that
each style requires, because naming a component is what makes the model render
it.

Vocabulary used deliberately
----------------------------
``pallu``
    The decorated loose end of a saree, draped over a shoulder.
``pleats`` / ``kunjalam``
    The fan of folds tucked at the front waist.
``choli`` / ``blouse``
    The fitted upper garment worn with a saree or lehenga.
``dupatta``
    The long scarf worn with a lehenga or salwar suit.
``ghagra`` / ``lehenga skirt``
    The flared floor-length skirt.
``churidar``
    Tightly fitted leggings gathered at the ankle, worn under a kurta.

Every template ends with the same preservation clause. Repeating it verbatim
matters: the clause is what stops the model regenerating the customer's face.
"""

from __future__ import annotations

from typing import Final

from ai_trial_room.config import Category, DrapeStyle, DupattaStyle

# --------------------------------------------------------------------------- #
# Shared clauses
# --------------------------------------------------------------------------- #

#: Identity-preservation clause appended to every prompt.
PRESERVE_CLAUSE: Final[str] = (
    "Keep the person's face, facial features, hairstyle, skin tone, body shape, "
    "height, pose and the background completely unchanged. Only replace the "
    "clothing."
)

#: Realism clause that pushes the model toward photographic output.
REALISM_CLAUSE: Final[str] = (
    "Photorealistic result with natural fabric folds, correct draping physics, "
    "soft shadows where the fabric meets the body, and lighting that matches "
    "the original photograph."
)

#: Fidelity clause: the garment must match the reference, not be reinvented.
FIDELITY_CLAUSE: Final[str] = (
    "Reproduce the garment from the second image exactly: the same colour, the "
    "same print and motif placement, the same border design, the same fabric "
    "texture and sheen, and the same embroidery or zari work."
)

#: Artefact and identity-drift terms shared by every negative prompt.
BASE_NEGATIVE: Final[str] = (
    "different face, different person, changed facial features, distorted face, "
    "extra limbs, extra arms, missing limbs, deformed hands, fused fingers, "
    "changed body proportions, changed background, blurry, low resolution, "
    "watermark, text, logo, oversaturated, plastic skin, cartoon, 3d render, "
    "mannequin, headless body"
)


# --------------------------------------------------------------------------- #
# Saree drape styles
# --------------------------------------------------------------------------- #


class DrapeSpec:
    """Physical description of one regional saree drape.

    Attributes
    ----------
    name:
        Display name used in the prompt.
    shoulder:
        Which shoulder the pallu falls over: ``"left"`` or ``"right"``.
    description:
        Sentence describing the pleats and pallu handling.
    region:
        Regional attribution, included to nudge the model's visual prior.
    avoid:
        Drape-specific failure modes, appended to the negative prompt. These are
        not generic quality terms - each one names the *specific* wrong drape
        this style tends to collapse into, which is what stops the collapse.
    pallu_focus:
        Short instruction used by the refinement pass when re-rendering just the
        pallu region.
    """

    def __init__(
        self,
        name: str,
        shoulder: str,
        description: str,
        region: str,
        avoid: str = "",
        pallu_focus: str = "",
    ) -> None:
        self.name = name
        self.shoulder = shoulder
        self.description = description
        self.region = region
        self.avoid = avoid
        self.pallu_focus = pallu_focus


DRAPE_SPECS: Final[dict[DrapeStyle, DrapeSpec]] = {
    DrapeStyle.NIVI: DrapeSpec(
        name="Nivi",
        shoulder="left",
        description=(
            "neat crisp pleats tucked in at the front waist and fanning downward, "
            "the pallu draped diagonally across the torso and falling over the "
            "left shoulder down the back, the midriff lightly visible above the "
            "waist"
        ),
        region="Andhra Pradesh, now the standard modern Indian drape",
        avoid=(
            "pallu over the right shoulder, pallu spread flat across the chest, "
            "no pleats at the waist, smooth wrapped fabric without pleat folds, "
            "saree worn like a dress or gown, bedsheet wrapped around the body"
        ),
        pallu_focus=(
            "the pallu falling over the left shoulder and down the back in soft "
            "vertical folds, its decorative border and end-piece design clearly "
            "visible"
        ),
    ),
    DrapeStyle.BENGALI: DrapeSpec(
        name="Bengali Atpoure",
        shoulder="left",
        description=(
            "no front pleats at the waist, instead wide box pleats at the sides, "
            "the pallu brought over the left shoulder, around the back and "
            "returned over the right shoulder to hang in front, often with a "
            "key-ring knotted into the end"
        ),
        region="Bengal",
        avoid=(
            "narrow fan of pleats at the front waist, Nivi style drape, pallu "
            "hanging only on one side, single shoulder drape, tight fitted wrap"
        ),
        pallu_focus=(
            "the wide pallu crossing both shoulders, one end returning to hang "
            "loose in front of the body, with visible border work"
        ),
    ),
    DrapeStyle.GUJARATI: DrapeSpec(
        name="Gujarati Seedha Pallu",
        shoulder="right",
        description=(
            "the pallu brought from the back over the RIGHT shoulder and spread "
            "open across the front of the chest so the decorated pallu design "
            "faces forward, with pleats tucked at the front waist"
        ),
        region="Gujarat and Rajasthan",
        avoid=(
            "pallu over the left shoulder, pallu hanging down the back, pallu "
            "folded into a narrow band, Nivi style diagonal drape, pallu behind "
            "the body, plain undecorated chest panel"
        ),
        pallu_focus=(
            "the pallu spread wide and flat across the chest with its full "
            "decorated design and border facing the camera"
        ),
    ),
    DrapeStyle.NAUVARI: DrapeSpec(
        name="Nauvari",
        shoulder="left",
        description=(
            "a nine-yard saree draped dhoti-style with the fabric passed between "
            "the legs and tucked at the back to form trouser-like folds, the "
            "pallu wrapped across the chest and over the left shoulder"
        ),
        region="Maharashtra",
        avoid=(
            "straight skirt, wrapped skirt, single tube of fabric around the legs, "
            "legs joined together under the fabric, regular Nivi saree, floor "
            "length skirt without leg separation, lehenga"
        ),
        pallu_focus=(
            "the pallu wrapped firmly across the chest and over the left "
            "shoulder, with the dhoti-style leg folds visible below"
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Category templates
# --------------------------------------------------------------------------- #

_SAREE_TEMPLATE: Final[str] = (
    "Dress the person in the first image in the saree shown in the second image, "
    "draped in the traditional {drape_name} style of {region}. "
    "The drape must have {drape_description}. "
    "Include a matching fitted blouse (choli) with elbow-length sleeves in a "
    "colour taken from the saree. The pallu falls over the {shoulder} shoulder. "
    "The saree hem reaches the ankles and breaks naturally over the feet. "
    "{fidelity} {realism} {preserve}"
)

_LEHENGA_TEMPLATE: Final[str] = (
    "Dress the person in the first image in the lehenga shown in the second "
    "image. Render the full three-piece outfit: a floor-length flared ghagra "
    "skirt that falls in wide gathers from the waist to the ankles, a fitted "
    "choli blouse, and {dupatta}. The waistline sits at the natural waist "
    "and the skirt flares outward with visible volume and pleat shadows. "
    "{fidelity} {realism} {preserve}"
)

#: How each dupatta style is described in the prompt.
_DUPATTA_DESCRIPTIONS: Final[dict[DupattaStyle, str]] = {
    DupattaStyle.SINGLE_SHOULDER: (
        "a dupatta draped over the {shoulder} shoulder with the loose end "
        "falling behind the arm in soft folds"
    ),
    DupattaStyle.BOTH_SHOULDERS: (
        "a dupatta draped symmetrically over both shoulders, its two ends "
        "hanging down the front on either side of the body"
    ),
    DupattaStyle.OVER_HEAD: (
        "a dupatta pinned at the crown of the head and falling over both "
        "shoulders in a bridal veil style, framing the face without covering it"
    ),
    DupattaStyle.ARM_DRAPE: (
        "a dupatta carried across both forearms in front of the body, held away "
        "from the torso so the choli and skirt stay fully visible"
    ),
}

_KURTI_TEMPLATE: Final[str] = (
    "Dress the person in the first image in the kurti shown in the second "
    "image. The kurti falls straight from the shoulders to mid-thigh or knee "
    "length, with a clean neckline, set-in sleeves and side slits that hang "
    "naturally. Keep the person's existing lower garment visible below the "
    "kurti hem. {fidelity} {realism} {preserve}"
)

_KURTA_TEMPLATE: Final[str] = (
    "Dress the person in the first image in the kurta shown in the second "
    "image. The kurta is a loose straight-cut tunic falling to the knees with "
    "a band collar or mandarin neckline, long sleeves, and a placket at the "
    "chest. It drapes loosely without clinging to the body. "
    "{fidelity} {realism} {preserve}"
)

_DRESS_TEMPLATE: Final[str] = (
    "Dress the person in the first image in the dress shown in the second "
    "image. Match the dress's silhouette, neckline, sleeve length and hem "
    "length exactly as shown in the reference. The fabric follows the person's "
    "body shape with natural drape and fold shadows. "
    "{fidelity} {realism} {preserve}"
)

_TEMPLATES: Final[dict[Category, str]] = {
    Category.SAREE: _SAREE_TEMPLATE,
    Category.LEHENGA: _LEHENGA_TEMPLATE,
    Category.KURTI: _KURTI_TEMPLATE,
    Category.KURTA: _KURTA_TEMPLATE,
    Category.DRESS: _DRESS_TEMPLATE,
}

#: Extra hint when the garment photo shows unstitched fabric rather than an
#: outfit being worn.
_UNSTITCHED_HINT: Final[str] = (
    "The second image shows the fabric as an unstitched length laid flat, not "
    "worn by anyone; drape it onto the person rather than copying its flat shape."
)

#: Extra hint when the garment reference is worn by a different model.
_WORN_HINT: Final[str] = (
    "The second image may show the garment on a different person or mannequin; "
    "transfer only the garment, never that person's face or body."
)

#: Framing hints so the model respects how much of the body is in shot.
_FRAMING_HINTS: Final[dict[str, str]] = {
    "full": "The person is shown full-length from head to feet; keep that framing.",
    "three_quarter": (
        "The person is visible from head to about the knees; do not extend the "
        "frame or invent the lower legs."
    ),
    "half": (
        "Only the upper body is visible; do not extend the frame or invent a "
        "lower body."
    ),
}


#: Category-specific failure modes, appended to the negative prompt.
#:
#: These are the mistakes each garment type actually collapses into - naming them
#: is far more effective than piling on generic quality adjectives.
_CATEGORY_NEGATIVES: Final[dict[Category, str]] = {
    Category.SAREE: (
        "western dress, gown, skirt and top, kurta, lehenga, bare midriff with no "
        "blouse, missing blouse, plain fabric with no border, jeans or trousers "
        "visible under the saree, bare legs, saree ending above the ankles"
    ),
    Category.LEHENGA: (
        "saree, single piece dress, gown, narrow straight skirt, pencil skirt, "
        "missing dupatta, missing choli, bare midriff with no blouse, trousers "
        "visible under the skirt, skirt with no flare or volume"
    ),
    Category.KURTI: (
        "saree, lehenga, gown, floor length dress, tight bodycon fit, missing "
        "sleeves where the reference has sleeves, kurti ending at the waist"
    ),
    Category.KURTA: (
        "saree, lehenga, women's blouse, tight fitted shirt, t-shirt, western "
        "suit jacket, kurta ending at the waist"
    ),
    Category.DRESS: (
        "saree, lehenga, kurta, separate skirt and top, changed neckline, changed "
        "sleeve length, changed hem length"
    ),
}

#: Prompt used when refining one region in a second pass.
_REFINE_TEMPLATE: Final[str] = (
    "Refine only the {region} of the {garment} this person is wearing. "
    "Render {focus}. Sharpen the woven border, zari thread, embroidery and print "
    "motifs so the fabric detail is crisp and the weave is visible. "
    "Keep the drape, silhouette, colour, pose, face and background exactly as "
    "they are - change nothing except the fidelity of the fabric detail."
)


def build_negative_prompt(
    category: Category,
    *,
    drape_style: DrapeStyle | None = None,
) -> str:
    """Compose the negative prompt for a category and drape.

    Layered from most general to most specific: shared artefact terms, then
    category confusions, then the exact wrong drape this style collapses into.

    Parameters
    ----------
    category:
        Target garment category.
    drape_style:
        Saree drape. Only contributes for :attr:`Category.SAREE`.

    Returns
    -------
    str

    Examples
    --------
    >>> negative = build_negative_prompt(Category.SAREE, drape_style=DrapeStyle.GUJARATI)
    >>> "pallu over the left shoulder" in negative
    True
    """
    parts = [BASE_NEGATIVE, _CATEGORY_NEGATIVES[category]]

    if category is Category.SAREE and drape_style is not None:
        spec = DRAPE_SPECS[drape_style]
        if spec.avoid:
            parts.append(spec.avoid)

    return ", ".join(part.strip().rstrip(",") for part in parts if part.strip())


def build_refine_prompt(
    category: Category,
    region_label: str,
    *,
    drape_style: DrapeStyle = DrapeStyle.NIVI,
) -> str:
    """Compose the instruction for a region-targeted refinement pass.

    Parameters
    ----------
    category:
        Garment category being refined.
    region_label:
        Human-readable region name, e.g. ``"pallu / dupatta"``.
    drape_style:
        Saree drape, which supplies the region-specific focus sentence.

    Returns
    -------
    str
    """
    if category is Category.SAREE:
        focus = DRAPE_SPECS[drape_style].pallu_focus or "the fabric in fine detail"
    elif category is Category.LEHENGA:
        focus = (
            "the dupatta and skirt fabric with crisp gathers and clearly defined "
            "border work"
        )
    else:
        focus = "the fabric weave, print motifs and stitched edges in fine detail"

    return _REFINE_TEMPLATE.format(
        region=region_label,
        garment=category.label.lower(),
        focus=focus,
    )


def build_prompt(
    category: Category,
    *,
    drape_style: DrapeStyle = DrapeStyle.NIVI,
    dupatta_style: DupattaStyle = DupattaStyle.SINGLE_SHOULDER,
    dominant_shoulder: str = "left",
    framing: str = "full",
    looks_unstitched: bool = False,
    extra: str = "",
) -> str:
    """Compose the instruction prompt for a reference-editing backend.

    Parameters
    ----------
    category:
        Target garment category.
    drape_style:
        Saree drape. Ignored for non-saree categories.
    dupatta_style:
        How a lehenga's dupatta is carried. Ignored for other categories.
    dominant_shoulder:
        Shoulder facing the camera, from pose detection. Used for lehenga
        dupatta placement and as a fallback for saree pallu side.
    framing:
        ``"full"``, ``"three_quarter"`` or ``"half"`` from
        :attr:`~ai_trial_room.preprocessing.person.PoseResult.framing`.
    looks_unstitched:
        True when the garment photo appears to show flat unstitched fabric.
    extra:
        Free-text instruction appended verbatim (from the UI).

    Returns
    -------
    str
        The complete prompt.

    Examples
    --------
    >>> p = build_prompt(Category.SAREE, drape_style=DrapeStyle.GUJARATI)
    >>> "RIGHT shoulder" in p
    True
    """
    template = _TEMPLATES[category]

    if category is Category.SAREE:
        spec = DRAPE_SPECS[drape_style]
        prompt = template.format(
            drape_name=spec.name,
            region=spec.region,
            drape_description=spec.description,
            shoulder=spec.shoulder,
            fidelity=FIDELITY_CLAUSE,
            realism=REALISM_CLAUSE,
            preserve=PRESERVE_CLAUSE,
        )
    elif category is Category.LEHENGA:
        dupatta = _DUPATTA_DESCRIPTIONS[dupatta_style].format(shoulder=dominant_shoulder)
        prompt = template.format(
            dupatta=dupatta,
            fidelity=FIDELITY_CLAUSE,
            realism=REALISM_CLAUSE,
            preserve=PRESERVE_CLAUSE,
        )
    else:
        prompt = template.format(
            fidelity=FIDELITY_CLAUSE,
            realism=REALISM_CLAUSE,
            preserve=PRESERVE_CLAUSE,
        )

    parts = [prompt, _FRAMING_HINTS.get(framing, "")]
    parts.append(_UNSTITCHED_HINT if looks_unstitched else _WORN_HINT)
    if extra.strip():
        parts.append(extra.strip())

    return " ".join(part for part in parts if part).strip()
