# Shop image sourcing — decide in the first hour (TODO(A))

No automatic download. Pick one path, write the result into `data/shop/raw/` (gitignored).

## Path 1: ImageNet validation images (recommended if access works)
- Gated `imagenet-1k` dataset on Hugging Face; each teammate must accept its terms with their own account.
- Take validation images for the 30 classes in `shop_config.yaml` (after `classes.py` resolves them).
- Hold out a disjoint set of clean images per class for the reference bank.

## Path 2: another licensed source
- Must allow this use; record the license and source URL here.
- Same layout: `data/shop/raw/<class_name>/<image_id>.jpg`.

Decision: _TBD_ · Licence notes: _TBD_
