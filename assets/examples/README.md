# Example images

Drop demo images here and the Gradio UI builds a one-click examples gallery
automatically (see `_example_rows` in [`app.py`](../../app.py)).

## Naming convention

The filename determines how an image is used:

```
person_01.jpg     person_02.jpg     …   ← model photos
dress_01.jpg      dress_02.jpg      …   ← garments, one prefix per category
kurti_01.jpg
kurta_01.jpg
saree_01.jpg
lehenga_01.jpg
```

Prefixes must match the `Category` enum values exactly: `dress`, `kurti`,
`kurta`, `saree`, `lehenga`. Any image extension Pillow reads will work.

## What makes a good example

**Person photos**

- Standing, facing the camera, arms away from the torso
- Even lighting, plain-ish background
- **Full length** (head to feet) - required for saree and lehenga examples, since
  `require_framing` rejects half-body photos for draped garments
- At least 384 px on the short edge

**Garment photos**

- Flat-lay or mannequin product shot, plain background
- The whole garment in frame - for a saree, include the pallu, which carries most
  of the design
- At least 256 px on the short edge

## Licensing and consent

`.gitignore` excludes this directory's contents by default, deliberately: photos
of people must not be committed by accident. To add a curated example:

```bash
git add -f assets/examples/person_01.jpg
```

Before you do, confirm:

- You have **written permission** from the person in the photo to publish it, or
  it is a stock/synthetic image licensed for redistribution.
- Garment photos are your own, or licensed for your use - shop catalogue images
  usually are not.

For a public demo, commissioning or generating a small set of consented images is
worth the effort. Scraping model photos from retail sites is both a copyright and
a personality-rights problem.
