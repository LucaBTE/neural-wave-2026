from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from PIL import Image
from studio_fidelity import context_reference, clean_detached_fragment


def test_context_crop_keeps_original_pixels_and_every_segmented_part():
    image = Image.new('RGB', (500, 800), (210, 180, 140))
    image.paste((120, 30, 60), (100, 200, 400, 700))
    mask = Image.new('L', image.size, 0)
    mask.paste(255, (100, 200, 400, 700))
    mask.paste(255, (60, 180, 75, 250))  # Part extending beyond the detector.
    target = {'bbox':[.2, .25, .8, .875], 'selection_validation':{'approved':True}}
    crop, bounds = context_reference(image, target, mask)
    assert bounds[0] <= 60 and bounds[1] <= 180
    assert bounds[2] >= 400 and bounds[3] >= 700
    assert crop.tobytes() == image.crop(bounds).tobytes()
    target['selection_validation']['approved'] = False
    original, bounds = context_reference(image, target, mask)
    assert bounds is None and original.tobytes() == image.tobytes()


def scene(background=(245,245,245)):
    image = Image.new('RGB', (500,800), background)
    image.paste((30,30,30), (100,300,400,700))
    image.paste((90,90,90), (180,100,320,160))
    image.paste((238,235,225), (80,240,90,320))  # Pale detail omitted by matte.
    mask = Image.new('L', image.size, 0)
    mask.paste(255, (100,300,400,700))
    mask.paste(255, (180,100,320,160))
    return image, mask


def test_cleanup_removes_only_detached_upper_fragment_and_keeps_rgb_parts():
    image, mask = scene()
    result, alpha, report = clean_detached_fragment(image, mask, 't-shirt')
    assert report['applied'] and len(report['removed_boxes']) == 1
    assert result.getpixel((250,130)) == (245,245,245)
    assert alpha.getpixel((250,130)) == 0
    assert result.crop((0,200,500,800)).tobytes() == image.crop((0,200,500,800)).tobytes()
    assert alpha.crop((0,200,500,800)).tobytes() == mask.crop((0,200,500,800)).tobytes()
