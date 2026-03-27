"""Recaption module – rewrite user instructions via a configurable LLM.

After N images have been grouped into 2-3 composite images, the original
references such as "图1", "图2", …, "image 3" no longer correspond to the
original individual images.  This module rewrites the user instruction so
that:

1. Image references point to the correct **composite** image.
2. The instruction is expanded to ~512 words / characters (matching the
   user's language) to provide richer context for FireRed-Image-Edit.
3. The user's original language is preserved (Chinese stays Chinese, etc.).

The LLM backend is selected via the ``RECAPTION_PROVIDER`` environment
variable.  Supported values: ``"gemini"`` (default),
``"openai_compatible"``, ``"minimax"``.  See :mod:`agent.llm_provider` for
details.
"""

from __future__ import annotations

import re
import time
import traceback
import warnings
from typing import Any

from agent.config import RECAPTION_TARGET_LENGTH

# Retry settings
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: float = 2.0  # seconds, doubled each retry


# ───────────────── simple reference mapping ──────────────────

def build_reference_map(
    group_indices: list[list[int]],
) -> dict[int, int]:
    """Create a mapping from **original** 1-based image number to the
    **composite** 1-based image number.

    Example
    -------
    >>> build_reference_map([[0, 1, 2], [3, 4]])
    {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    """
    mapping: dict[int, int] = {}
    for group_idx, members in enumerate(group_indices):
        for orig_idx in members:
            mapping[orig_idx + 1] = group_idx + 1  # 1-based
    return mapping


def _replace_image_refs(
    text: str,
    ref_map: dict[int, int],
) -> str:
    """Replace image-N references in *text* according to *ref_map*.

    Handles patterns like:
    - 图1 / 图2 / 图N  (Chinese)
    - image 1 / Image 2 / Image N  (English)
    - img1 / IMG 3  (abbreviated)
    - 第1张图 / 第2张图  (Chinese ordinal)
    """

    def _sub_zh(m: re.Match) -> str:
        num = int(m.group(1))
        new_num = ref_map.get(num, num)
        return f"图{new_num}"

    def _sub_en(m: re.Match) -> str:
        num = int(m.group(2))
        new_num = ref_map.get(num, num)
        return f"{m.group(1)}{new_num}"

    def _sub_ordinal(m: re.Match) -> str:
        num = int(m.group(1))
        new_num = ref_map.get(num, num)
        return f"第{new_num}张图"

    text = re.sub(r"图(\d+)", _sub_zh, text)
    text = re.sub(r"((?:image|img|IMAGE|IMG)\s*)(\d+)", _sub_en, text, flags=re.IGNORECASE)
    text = re.sub(r"第(\d+)张图", _sub_ordinal, text)
    return text


# ───────────────── Gemini-powered recaption ──────────────────

_SYSTEM_PROMPT = """\
# Edit Instruction Rewriter

You are an expert at refining image-editing instructions. Given the user's \
original prompt and (optionally) a mapping of how the source images were \
regrouped into composite images, produce a single improved editing \
instruction that a diffusion-based editing model can follow precisely.

## Core Rules

1. **Preserve intent.** Never invent new operations. Only clarify, enrich, \
and correct what the user already asked for.
2. **Stay in the user's language.** If the input is Chinese, output Chinese. \
If English, output English. Never translate.
3. **Target length ≈ {target_len} words (English) or characters (CJK).** \
Keep sentences tight—prefer specificity over verbosity.
4. **Image references.** When a regrouping mapping is provided, rewrite \
every mention of "image 1 / 图1 / 第1张图 / img1" etc. to match the new \
composite numbering.
5. **Output the rewritten instruction only.** No preamble, no explanation.

## How to Handle Different Edit Types

### Object Addition / Removal / Replacement
- If the instruction already specifies entity, position, quantity, and \
attributes clearly, just polish the grammar.
- If it is vague, fill in the minimum necessary details—object category, \
colour, rough size, orientation, and placement—so the result is \
unambiguous. E.g.  \
  "Add an animal" → "Place a small ginger tabby cat sitting in the \
lower-right corner, facing the viewer."
- For replacements, use the pattern "Replace ⟨old⟩ with ⟨new⟩" and \
briefly note key visual traits of the replacement.
- Drop nonsensical operations (e.g. "add 0 items").

### Text Overlay / Text Replacement
- Wrap every piece of text content in English double-quotes `" "`. \
Preserve the original language and casing of the text itself.
- Use the template: `Replace "old text" with "new text"` or \
`Add text "CONTENT" at ⟨position⟩ in ⟨style⟩`.
- If no text content is given, infer something short and contextually \
appropriate from the scene.

### Person / Portrait Editing
- Retain identity-critical attributes (ethnicity, gender, age group, \
hairstyle, expression, attire) unless the edit explicitly changes them.
- Appearance modifications (clothing, accessories) should be stylistically \
coherent with the rest of the image.
- Expression adjustments must stay natural—avoid cartoon-like exaggeration.
- For background swaps, lead with "Keep the subject unchanged; …" to \
anchor the person's appearance.

### Style Transfer / Enhancement / Restoration
- Summarise the target style with 3–5 distinctive visual cues (palette, \
lighting, texture, era, medium). E.g.  \
  "Disco style" → "Retro 1970s disco atmosphere—warm neon palette, \
mirror-ball reflections, soft lens flare, saturated tones."
- For old-photo restoration or colourisation, begin with: \
"Restore and colourise the photograph: remove scratches, reduce grain, \
recover natural skin tones, sharpen facial details, maintain vintage \
warmth."
- Append style notes at the end when other edits are also requested.

## Sanity Checks
- If the instruction contains contradictions, resolve them with the most \
reasonable interpretation and note the fix inline.
- If spatial placement is missing, pick a compositionally balanced location \
(near the subject, in negative space, along rule-of-thirds lines, etc.).\
"""


def recaption(
    original_instruction: str,
    group_indices: list[list[int]],
    target_length: int = RECAPTION_TARGET_LENGTH,
) -> str:
    """Rewrite *original_instruction* with updated image references and richer
    detail.

    Parameters
    ----------
    original_instruction:
        The user's raw editing prompt (may be Chinese, English, or mixed).
    group_indices:
        Output of :func:`agent.image_tools.build_group_mapping` – each element
        is a list of 0-based original image indices that have been stitched
        into one composite image.
    target_length:
        Desired length of the rewritten instruction (words or characters).

    Returns
    -------
    The rewritten instruction string.
    """
    ref_map = build_reference_map(group_indices)

    # First do a deterministic regex pass to fix obvious references
    instruction_fixed = _replace_image_refs(original_instruction, ref_map)

    # Build the LLM prompt
    mapping_desc = {
        f"original_image_{orig}": f"composite_image_{comp}"
        for orig, comp in ref_map.items()
    }

    user_prompt = (
        f"Original instruction:\n{instruction_fixed}\n\n"
        f"Image grouping mapping (original image number → composite image number):\n"
        f"{mapping_desc}\n\n"
        f"Please output the rewritten instruction (approximately {target_length} words/characters)."
    )

    try:
        from agent.llm_provider import get_recaption_provider

        provider = get_recaption_provider()
    except (ImportError, ValueError) as exc:
        warnings.warn(
            f"[Recaption] Failed to initialise LLM provider: {exc!r}. "
            "Skipping recaption and using the original prompt.",
            RuntimeWarning,
            stacklevel=2,
        )
        return instruction_fixed

    system_prompt = _SYSTEM_PROMPT.format(target_len=target_length)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            rewritten = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=4096,
            )
            return rewritten if rewritten else instruction_fixed
        except Exception as exc:
            last_exc = exc
            traceback.print_exc()
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"[Recaption] Attempt {attempt}/{_MAX_RETRIES} failed, "
                    f"retrying in {wait:.0f}s …"
                )
                time.sleep(wait)

    # All retries exhausted – warn and fall back to the original prompt.
    warnings.warn(
        f"[Recaption] LLM call failed after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc!r}. "
        f"Falling back to the original prompt for inference.",
        RuntimeWarning,
        stacklevel=2,
    )
    return instruction_fixed
