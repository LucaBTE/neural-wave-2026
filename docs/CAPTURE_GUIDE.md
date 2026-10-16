# Capture guide, hints for the on-site set

Optional capture guidance. Designing the capture protocol is part of the challenge: a team that
captures cleverly with a simple pipeline can beat a sophisticated one fed bad inputs. This guide
lists the things that will otherwise cost you points for reasons that have nothing to do with your
pipeline.

You are scored on flat-lay packshots of the 15 on-site garments. Your inputs are your own photos,
shot on the day with the set Sartiq provides (backdrop and one controllable light) and your own
smartphone, or the phone photos Sartiq supplies. Both are allowed. Each garment carries a tag with
its id, g01 to g15.

---

## The two things that must be honest

The challenge is explicit: the two things that genuinely have to be captured honestly are light
and colour. Avoid clipped highlights and inconsistent white balance. Everything below is in
service of that.

---

## 1. Set up the geometry first, then light

* Shoot down, not across. Lay the garment flat on the backdrop and put the phone on a tripod or a
  stack of boxes directly above it, lens parallel to the fabric. A hand-held angled shot makes
  perspective remove the shape for you, and the silhouette term (mask IoU, 15%) will notice.
* Fill the frame with the garment, whole. The scorer crops to the mask and rescales, so margins
  cost you nothing, but a clipped sleeve or hem is a real error (`truncated` becomes `review`,
  and the mask bbox is wrong). Leave a comfortable border.
* Smooth the garment out before shooting: sleeves symmetric, not folded over itself, nothing else
  in frame. The scorer keeps every mask component of 0.5% of the image or more, so a stray corner
  of a hanger becomes part of the garment you are scored on.
* Height is free, resolution is not. Get the garment large in the sensor. You need at least 256 px
  on the short side, and you want detail for LPIPS.

## 2. Control the light

* One soft, even light beats two hard ones. Diffuse the provided light (or bounce it off white
  card) and put it at about 45 degrees from above. A raw point light creates a hot spot and a
  hard shadow that reads as the wrong material.
* Kill your own shadow. Step out of the frame, or tape the light down and use a timer or a remote
  shutter. A phone-shaped shadow over the garment is unrecoverable detail.
* Avoid mixed colour temperature. Daylight through a window plus a warm lamp makes white balance
  unsolvable for a single number. Close the blinds or work away from the window.
* Do not overexpose. Blown highlights lose the fabric texture that LPIPS and SSIM measure, and
  they clip the colour you are scored on. Check the histogram, not the phone screen.

## 3. Get colour referenceable

* Include a white or grey reference in every frame (a grey card, a sheet of print paper, the
  inside of the backdrop roll) and keep it in a fixed corner. Whether you crop it out afterwards
  or use it to compute a per-shot white balance is your decision. If you do neither, you are
  gambling on the phone's auto white balance, which drifts shot to shot.
* Lock exposure and white balance manually if your camera app allows it. Auto-WB hunting between
  two frames of the same garment is a guaranteed inconsistency penalty.
* Shoot RAW or the highest-quality JPEG you can with a fixed WB. Re-encode to PNG for submission.
* Test on the hardest garment you have first: white, cream, lace, thin straps. Those are the cases
  where the segmenter and your lighting both fail first.

## 4. Angles

* The required view is the front. Photograph the front flat while the garment is laid out.
* Capture as many angles as your pipeline wants (multi-view fusion, relighting); only the front is
  scored against the reference.
* A hanger or dress-form capture is faster than flat-lay and may be enough for your pipeline to infer
  the shape. Remember you are scored against a flat-lay reference, so anything the pipeline has to
  undo (hanger hook, folds, gravity) is error it has to fix. Test both styles on your holdout
  before committing your on-site time to the faster one.

## 5. What to actually do in the ten minutes before you shoot

1. Put the garment down, smooth it, fill the frame, leave a margin.
2. Light at about 45 degrees, diffused, on the far side of your phone so your shadow falls away.
3. Grey card in frame, exposure locked, WB locked, blinds closed.
4. Shoot the front, check the histogram. Shoot more angles only if your pipeline uses them.
5. Copy to a machine, run your pipeline, and compare the output to the garment, not to the phone
   photo. Fix the capture before you fix the pipeline.

## 6. Anti-patterns that lose points for free

* Garment overlapping the backdrop edge or a table edge: weird mask, `truncated`.
* Dark garment on dark backdrop, or white on white: the segmenter's weak spot.
* A stack of other garments in frame: extra mask components.
* Shooting through a phone screen preview of the previous shot.
* Re-shooting later. Capture all 15 garments cleanly the first time. Improving a good capture costs
  minutes; recovering a bad one costs the garment.
