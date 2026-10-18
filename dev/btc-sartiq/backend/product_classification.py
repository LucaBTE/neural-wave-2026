"""Locate the intended product before segmentation; never default to clothing."""
import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path

from vision import DEFAULT_MODEL, CACHE_TTL, CACHE_SIZE, prepare_multiview_analysis
from request_reuse import ReuseCache

CLASSIFICATION_MODEL = DEFAULT_MODEL
LOCALIZATION_MODEL = 'fal-ai/moondream3-preview/detect'
PROMPT_PATH = Path(__file__).with_name('prompts') / 'classification.txt'
_cache = ReuseCache(ttl=CACHE_TTL, max_entries=CACHE_SIZE)
_location_cache = ReuseCache(ttl=CACHE_TTL, max_entries=CACHE_SIZE)


def clear_classification_cache():
    _cache.clear()
    _location_cache.clear()


def physical_category_kind(category):
    """Resolve clear physical noun heads, never OCR/brand words in the label.

    Whole noun heads avoid treating 'dress shoes' as a dress or a 'T-shirt bag'
    as clothing. Qualifiers after 'with' describe parts of the selected product.
    Unknown categories retain the model's routing instead of defaulting to clothes.
    """
    head = re.split(r'\s+(?:with|featuring|and)\s+|,', category.casefold(), maxsplit=1)[0].strip()
    groups = {
        'garment': (r'dress(?:es)?|gowns?|shirts?|t[ -]?shirts?|tees?|tops?|blouses?|'
                    r'sweaters?|jumpers?|pullovers?|cardigans?|hoodies?|sweatshirts?|'
                    r'skirts?|trousers?|pants|jeans|shorts|leggings|jackets?|coats?|'
                    r'vests?|waistcoats?|blazers?|bodysuits?|rompers?|jumpsuits?|'
                    r'overalls|dungarees|swimsuits?|garments?|clothing|apparel'),
        'accessory': (r'shoes?|boots?|sandals?|sneakers?|bags?|handbags?|backpacks?|'
                      r'purses?|belts?|hats?|caps?|scarves|scarf|ties|neckties|'
                      r'sunglasses|glasses|eyewear|watches|watch|jewelry|jewellery'),
        'other': r'rugs?|carpets?|blankets?|towels?|curtains?|fabric|textile items?',
    }
    for kind, nouns in groups.items():
        if re.search(r'\b(?:' + nouns + r')$', head):
            return kind
    return None


def parse_target(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.IGNORECASE)
    try:
        target = json.loads(text)
        if not isinstance(target, dict):
            raise ValueError('Expected an object')
        label, kind = target['label'], target.get('kind', 'other')
        if not isinstance(label, str):
            raise ValueError('Invalid product label')
        label = ' '.join(label.split())
        if (not 1 <= len(label) <= 120 or not any(char.isalpha() for char in label) or
                any(not char.isprintable() for char in label) or
                label.casefold() in ('unknown', 'none', 'null', 'n/a', 'unclear', 'ambiguous', 'not sure')):
            raise ValueError('No usable product name')
        unknown_kind = not isinstance(kind, str) or 'kind' not in target
        kind = kind.strip().lower() if isinstance(kind, str) else 'other'
        kind = {'clothing': 'garment', 'apparel': 'garment', 'accessories': 'accessory',
                'object': 'other', 'product': 'other'}.get(kind, kind)
        unknown_kind = unknown_kind or kind not in ('garment', 'accessory', 'other')
        if unknown_kind:
            kind = 'other'  # Never assume garment styling for an uncertain object.
        # A brand or OCR result is not a physical segmentation target. Keep the
        # category separate so text such as "Jennie Ruby" cannot locate a print.
        category = target.get('category')
        if not isinstance(category, str):
            raise ValueError('Missing physical product category')
        category = ' '.join(category.split())
        # Some vision responses echo the schema's type description verbatim.
        # A detector queried with "physical noun" can return a print or nothing.
        # Recover only an explicit product noun, never OCR/brand names.
        if category.casefold() in ('physical noun', 'physical item', 'physical product',
                                   'product', 'object', 'item', 'garment', 'clothing', 'apparel'):
            recovered = concrete_category(label, target.get('description'))
            if recovered:
                category = recovered
            elif category.casefold() not in ('garment', 'clothing', 'apparel'):
                raise ValueError('Unresolved category placeholder')
        if (not 1 <= len(category) <= 60 or not any(c.isalpha() for c in category) or
                any(not c.isprintable() for c in category) or
                category.casefold() in ('unknown', 'none', 'null', 'other', 'print',
                                        'graphic', 'logo', 'text', 'portrait', 'image',
                                        'design', 'artwork', 'brand')):
            raise ValueError('Invalid physical product category')
        model_kind = kind
        inferred_kind = physical_category_kind(category)
        if inferred_kind is not None:
            kind, unknown_kind = inferred_kind, False
        confidence = target.get('confidence')
        if isinstance(confidence, str):
            try:
                confidence = float(confidence.strip())
            except ValueError:
                confidence = None
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            confidence = None  # Missing/invalid metadata must not invent certainty.
        review = unknown_kind or confidence is None or confidence < .6
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Product classification returned no usable product identity. Try a clearer photo.') from exc
    # Text-generated coordinates must never guide segmentation.
    result = {'label': label, 'category': category, 'kind': kind,
              'confidence': confidence, 'review_required': review}
    if kind != model_kind:
        result.update(model_kind=model_kind, kind_source='physical_category')
    if 'short english product name' in label.casefold() or 'distinguishing visual attributes' in label.casefold():
        result.update(label=category, confidence=None, review_required=True, identity_supported=False)
    description = target.get('description')
    if (isinstance(description, str) and description.strip() and
            not any(phrase in description.casefold() for phrase in
                    ('short english product name', 'distinguishing visual attributes'))):
        result['description'] = ' '.join(description.split())[:600]
    return result


def concrete_category(label, description=None):
    """Recover the subject noun from a label or the subject of its caption.

    Explicit non-generic categories are never overridden. Stop before descriptions
    of artwork, parts or the surrounding scene, which may name other products.
    """
    for text in (label, description):
        if not isinstance(text, str):
            continue
        subject = re.split(r'\b(?:with|featuring|depicting|showing|printed|on|behind|beside)\b',
                           text.casefold(), maxsplit=1)[0].strip(' ,.')
        if re.search(r'\b(?:print|image|picture|portrait|logo|graphic|text)\b', subject):
            continue
        words = subject.split()
        # Last noun head handles compounds such as "dress shoes" and "T-shirt bag".
        for end in range(len(words), 0, -1):
            phrase = ' '.join(words[:end]).strip(' ,.')
            if physical_category_kind(phrase) is not None:
                noun = words[end - 1].strip(' ,.')
                if noun == 'shirt' and end > 1 and words[end - 2] == 't':
                    noun = 't-shirt'
                return noun
    return None


def detection_box(response):
    if not isinstance(response, dict):
        raise RuntimeError('Product detection returned an invalid response')
    payload = response.get('data') if isinstance(response.get('data'), dict) else response
    if payload.get('finish_reason') not in (None, 'stop'):
        raise RuntimeError('Product detection did not finish normally')
    objects = payload.get('objects')
    if not isinstance(objects, list):
        raise RuntimeError('Product detection did not return object boxes')
    candidates = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        box = [obj.get(key) for key in ('x_min', 'y_min', 'x_max', 'y_max')]
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in box):
            continue
        left, top, right, bottom = box
        cx, cy = (left + right) / 2, (top + bottom) / 2
        if not (left < right and top < bottom and .001 <= (right-left)*(bottom-top) <= .95):
            continue
        if not (.15 <= cx <= .85 and .15 <= cy <= .85):
            continue
        # A distant shirt on a side rack must not pass merely because its box
        # center barely clears the outer margin. Require actual overlap with the
        # central product area; this still permits hollow/off-center eyewear and
        # full-length garments, without forcing a foreground point into the mask.
        central_overlap = (max(0., min(right, .70) - max(left, .30)) *
                           max(0., min(bottom, .80) - max(top, .25)))
        if central_overlap / ((right - left) * (bottom - top)) < .25:
            continue
        candidates.append(((cx-.5)**2 + (cy-.5)**2, box))
    if not candidates:
        raise RuntimeError('Could not locate the classified product in the image')
    return min(candidates, key=lambda candidate: candidate[0])[1]


def locate_product(image, target, run_model):
    identity = {key: value for key, value in target.items()
                if key not in ('bbox', 'point', 'localization_cache_hit')}
    preview = image.convert('RGB')
    preview.thumbnail((1280, 1280))
    key = hashlib.sha256(LOCALIZATION_MODEL.encode() + target['category'].encode() +
                         str(preview.size).encode() + preview.tobytes()).hexdigest()
    def compute():
        encoded = io.BytesIO()
        preview.save(encoded, format='JPEG', quality=95, subsampling=0, optimize=True)
        response = run_model(LOCALIZATION_MODEL, {
            'image_url': 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii'),
            'prompt': target['category'], 'preview': False})
        return detection_box(response)
    box, reuse = _location_cache.get_or_compute(key, compute)
    return {**identity, 'bbox': box, 'localization_cache_hit': reuse != 'computed',
            'localization_cache_status': reuse}


def classify_product(image, run_model, extract_text, additional_images=None):
    prompt = PROMPT_PATH.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Product classification prompt must not be empty')
    extra = additional_images or []
    if len(extra) > 4:
        raise ValueError('Maximum 4 additional views')
    preview = image.convert('RGB')
    if extra:
        preview = prepare_multiview_analysis(preview, extra, 1280)
        prompt += ('\nThese panels show ONE product. Main may be a cropped detail or label. '
                   'Use the most complete view to identify the whole item; never classify '
                   'a seam, shoulder, printed image or care label as the product. '
                   'Describe only details supported by these photographs.')
    preview.thumbnail((1280, 1280))
    key = hashlib.sha256(CLASSIFICATION_MODEL.encode() + prompt.encode() +
                         str(preview.size).encode() + preview.tobytes()).hexdigest()
    def compute():
        encoded = io.BytesIO()
        preview.save(encoded, format='JPEG', quality=95, subsampling=0, optimize=True)
        response = run_model(CLASSIFICATION_MODEL, {
            'image_url': 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii'),
            'prompt': prompt, 'reasoning': False, 'temperature': 0})
        text = extract_text(response)
        payload = response.get('data') if isinstance(response.get('data'), dict) else response
        if payload.get('finish_reason') not in (None, 'stop'):
            raise RuntimeError('Product classification did not finish normally')
        return parse_target(text)
    target, reuse = _cache.get_or_compute(key, compute)
    return {**target, 'cache_hit': reuse != 'computed', 'cache_status': reuse}
