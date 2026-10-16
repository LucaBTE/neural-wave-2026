import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from PIL import Image
from pipeline import background_mask,extract_urls

def test_mask_inverts_and_protects_garment():
    m=Image.new('L',(20,20),0)
    for x in range(5,15):
        for y in range(5,15): m.putpixel((x,y),255)
    bg=background_mask(m)
    assert bg.getpixel((10,10))==0
    assert bg.getpixel((0,0))==255

def test_sam_masks_and_generation_response():
    assert extract_urls({'masks':[{'url':'https://a'},{'url':'https://b'}]},masks=True)==['https://a','https://b']
    assert extract_urls({'images':[{'url':'https://out'}]})==['https://out']
