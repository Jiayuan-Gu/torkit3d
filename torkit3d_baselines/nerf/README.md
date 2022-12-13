# NeRF: Neural Radiance Fields

## Installation

```bash
# pytorch-lightning==1.8.3.post1
pip install pytorch-lightning torchmetrics
```

## Data

```bash
mkdir -p data
cd data
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
```

## Reference

- Official implementation (Tensorflow): <https://github.com/bmild/nerf>
- Pytorch implementation: <https://github.com/yenchenlin/nerf-pytorch>
- Keras tutorial: <https://keras.io/examples/vision/nerf/>
- Pytorch3D tutorial: <https://pytorch3d.org/tutorials/fit_simple_neural_radiance_field>

**Difference between my implementation and the official one**:

- query points and segment lengths
- I do not implement deterministic stratified sampling (`sample_pdf`). Note that it seems that `perturb==0` will never be true in the original implementation since perturb is set to `False` instead of `0` during inference.
- I do not split query points into chunks to reduce memory footprint.
- The background color is processed in a slightly different way.
- I have not implemented `raw_noise_std`.
