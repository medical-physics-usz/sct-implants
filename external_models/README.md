# External models

We use exact snapshots of pix2pix and CUT. Their original
`README.md` and `LICENSE` files are kept intact. Thesis-specific changes are noted here:

**Note:** The modified classes, functions and lines have been marked with a `[thesis]` tag in the corresponding files.

---

## pix2pix and CycleGAN

### Snapshot
-  Repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
-  Upstream Commit: `c3268edd50ec37a81600c9b981841f48929671b8` (2024-03-22)

### Modified files

- `data/`
  - `aligned_implant_dataset.py` has been created based on original `aligned_implant.py` to enable compatibility
  with NIFTI files instead of PNGs. It ensures that all the required input and output images can be loaded 
  (different MR reconstructions, metal masks, CT).
  - `image_folder.py` has been modified such that NIFTI files can be loaded.
  
- `models/`
  - `networks.py` contains implementations of SPADE, MGA and weighted L1-loss. 
    - `SPADE`, `SPADEResNetBlock` and `ResnetGeneratorWithSPADE` classes have been added based on: [SPADE repository](https://github.com/NVlabs/SPADE/tree/master).
    - `ResnetGeneratorWithAttention`, `MetalGuidedAttention` classed have been added.
    - `RegionWeightedL1Loss` class has been added.
  - `pix2pix_model.py` has been modified to load metal masks from dataloader, as well as handling
  SPADE, MGA, and weighted L1-loss adaptations.

- `options/`
  - `base_options.py` has been extended with certain arguments. For example `--input_modalities` and `--output_modalities`
  specify on which input and output modalities the models should be trained on.
  - `test_options.py` has been extended with certain arguments. 

- `util/`
  - `visualizer.py` has been modified in order to display multiple input channels (eg. different MR inputs)
  individually.
  
- `train.py` has been slightly adapted to configure input and output channels correctly.
- `test.py` has been slightly adapted to save the generated synthetic CTs to NIFTI files.


---
## CUT

### Snapshot
-  Repo: https://github.com/taesungp/contrastive-unpaired-translation  
-  Upstream Commit: `b3ac297708dfb6f7589d04662277e53c0d579c27` (2023-09-05)

### Modified files

- `data/`
  - `aligned_implant_dataset.py` has been created based on original `aligned_implant.py` to enable compatibility
  with NIFTI files instead of PNGs. It ensures that all the required input and output images can be loaded 
  (different MR reconstructions, metal masks, CT).
  - `image_folder.py` has been modified such that NIFTI files can be loaded.

- `options/`
  - `base_options.py` has been extended with certain arguments. For example `--input_modalities` and `--output_modalities`
  specify on which input and output modalities the models should be trained on.
  - `test_options.py` has been extended with certain arguments.
  
- `train.py` has been slightly adapted to configure input and output channels correctly.
- `test.py` has been slightly adapted to save the generated synthetic CTs to NIFTI files.







