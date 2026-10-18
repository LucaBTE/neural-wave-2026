# Pipeline 0.1

| Step | Model or library | Purpose |
|---|---|---|
| Decode | Pillow / ImageCms | Apply EXIF orientation and convert embedded profiles to sRGB. |
| Identify | `fal-ai/moondream3-preview/query` | Identify and describe the foreground product; extra views share one local board. |
| Locate | `fal-ai/moondream3-preview/detect` | Locate that physical product rather than a print or background clothing. |
| Segment | `fal-ai/sam-3/image` | Select the complete product and check agreement with localization. |
| Prepare references | Pillow | Crop original RGB with 8% context when selection is supported; deduplicate photos. |
| Generate | `google/nano-banana-2.1/edit`, 2K | Produce one frontal, pressed flat lay, preserving the item's colors, graphics and parts. |
| Check | `fal-ai/moondream3-preview/query` | Check for duplicates. A detected duplicate gets one corrective edit and a mandatory recheck. |
| Finish | `fal-ai/birefnet/v2`, Pillow | Obtain a diagnostic matte and pad generated RGB to 1200 × 1800, preserving pale details and shadows. |
| Refine colors | OpenCV / NumPy | Apply the unchanged bounded studio-neutrality/material correction and conservative fragment cleanup. |

LangGraph orchestrates these steps. Normal execution uses six model inferences;
cached preprocessing reduces this to three. Duplicate correction adds at most two.
There is no separate garment-analysis call or second steaming pass.

Unusable classification, localization or masks fall back to the original source
photographs. A missing matte preserves the generated image. An inconclusive count
check marks the result for review; a known duplicate must pass rechecking before
being delivered. Generative editing can still change details or miss a duplicate.

Editable prompts are in `dev/btc-sartiq/backend/prompts/`. The generation template
keeps `{{LIGHTING_INSTRUCTIONS}}` and `{{GARMENT_DESCRIPTION}}` exactly once.
Descriptions are auxiliary data; photographs establish product colors.

The API contract, progress events, intermediate previews and model/library traces
remain available to the existing frontend. `quality` contains internal diagnostics,
not the official competition score.
