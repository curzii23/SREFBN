# SREFBN

This is the official repository for Super Resolution via Enchanced Feature Block Network (SREFBN). The pretrained models are given.

## Requriments

- Python 3.8.5
- Numpy 1.19.1
- Pytorch 1.6.0
- Windows 10

## Training

The training dataset can be downloaded from [Training dataset DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
* Run training x2, x3, x4 model
```bash
python train.py --root training_data/ --depth=48 --scale 2 --pretrained pretrained/48/srefpn_x2.pth
python train.py --root training_data/ --depth=48 --scale 3 --pretrained pretrained/48/srefpn_x3.pth
python train.py --root training_data/ --depth=48 --scale 4 --pretrained pretrained/48/srefpn_x4.pth
```
The tree structure of the training data should be:

```bash
training_data
├── DIV2K_decoded
     └──DIV2K_HR
     └──DIV2K_LR_bicubic
         └──x2
         └──x3
         └──x4
```

## Testing

Download the test datasets from [Testing Datasets](https://1drv.ms/u/s!AkqWwiuX5ZbigRDvouGNzxi9LPYG?e=Gjxdus). After extracting, the folder should be placed within the main directory in order to run the below commands.
* Run testing x2, x3, x4 models on Set5, Set14, BSDS100, Urban100 and Manga109.
```bash
python test.py --upscale_factor 2 --depth 48 --checkpoint pretrained/48/srefpn_x2.pth
python test.py --upscale_factor 3 --depth 48 --checkpoint pretrained/48/srefpn_x3.pth
python test.py --upscale_factor 4 --depth 48 --checkpoint pretrained/48/srefpn_x4.pth
```
## Citation

If you find the paper useful, please cite our paper from https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12546.
