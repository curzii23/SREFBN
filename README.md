# SREFPN

This is the official repository for Super Resolution via Enchanced Feature Pyramid Network (SREFPN).


* Run training x2, x3, x4 model
```bash
python train.py --root /path/to/DIV2K_decoded/ --scale 2 --pretrained pretrained/48/epoch_730_x2.pth
python train.py --root /path/to/DIV2K_decoded/ --scale 3 --pretrained pretrained/48/epoch_786_x3.pth
python train.py --root /path/to/DIV2K_decoded/ --scale 4 --pretrained pretrained/48/epoch_772_x4.pth
```

* Run testing x2, x3, x4 models on Set5, Set14, BSDS100, Urban100 and Manga109. The models will produce the accuracies equivalent to mentioned in the paper.
```bash
python test.py --upscale_factor 2 --depth 48 --checkpoint pretrained/48/epoch_730_x2.pth
python test.py --upscale_factor 3 --depth 48 --checkpoint pretrained/48/epoch_786_x3.pth
python test.py --upscale_factor 4 --depth 48 --checkpoint pretrained/48/epoch_772_x4.pth
```

## Citation

If you find SREFPN useful in your research, please consider citing:
