# torch-es
Double Seasonal Exponential Smoothing using [`PyTorch`](https://pytorch.org) with batched data and multiple series training support.


# 📋 Roadmap

There are lots of tools built on top of the code in this repository, so the plan is to add them here eventually.

Here's what's published:

- [x] 3d Holt-Winters implementation
- [x] Additive and Multiplicative seasonalities
- [x] Blender module to merge predictions from multiple series.
- [ ] Training loop for normal and bptt training.
- [ ] Uncertainty estimation via sampling.
- [ ] Additional losses
- [ ] RNN training on top of HW.

# 📚 Dependencies

- torch
- numpy
