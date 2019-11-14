# torch-es
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Using PyTorch](https://img.shields.io/badge/PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/vshulyak/torch-es/blob/master/LICENSE)

Double Seasonal Exponential Smoothing using [`PyTorch`](https://pytorch.org) with batched data and multiple series training support.*

# 📋Intro

After the M4 Competition, I really wanted to implement Slawek's approach in Pytorch and adapt it to my purposes. The reason:
I had quite good results with double seasonal HW models for my use case (short-term forecasting), but it was a bit awkward
to train multiple models to integrate exog effects.

This repo contains:
- [x] 3d Holt-Winters implementation (=multiple series processing in one step).
- [x] Additive and Multiplicative seasonalities.
- [x] LSTM residuals learner.
- [x] Exog predictors learner.
- [x] Autoregressive learner.
- [x] Blender module to merge predictions from multiple series.
- [x] Simple Quantile loss to get prediction intervals.
- [x] Training loop (no bptt, didn't prove to be useful for my case).

This is just part of the code which proved to be at least somewhat useful.

# Differences Slawek's version and this one

I never had an idea to reproduce the complete M4 prediction results (although it's a nice approach to validate the algorithm).

Here are the differences:
- Double seasonalities support.
- Just like in classical HW, seasonality changes with every update (on every step). In some cases in might help the residual learner (LSTM) to learn remaining info, in some it will cause problems.
- Seasonalities are treated as kind of embeddings, with custom length (additional holidays can be added / some special days)

# Lessons Learned

In hindsight, it's a good effort :) But there are other much better/easier ways to get good point forecasts and prediction
intervals. Namely, Kalman Filters/recursive online learning/reservoir computing made the results much better for my case.
Now I just need merge this all in one API.

Hence, I'm working on a second iteration of this module which will contain something more advanced and elaborated.

# The API

```
loss_fn = nn.MSELoss()

phases = [
    Phase(name='train', is_training=True, dataloader=train_dl),
    Phase(name='test', is_training=False, dataloader=test_dl),
]

model = PipelineModel([
    ExogConvLearner(cnt_input_dim=1),
    DSHWAdditiveLearner(period1_dim=24, period2_dim=24*8, h=24, enable_trend=False,
                        enable_main_loss=True,
                        enable_seas_smoothing=True
                       ),
    ARLearner(),
    OneStepLSTMLearner(cnt_input_dim=1),
    PredictionQuantileLearner()
], loss_fn=loss_fn)
model.apply(weights_init)

optimizer = AdamW(setup_model_params(model), lr=0.01 , weight_decay=1e-6)

state_dicts = train(model, phases, opt=optimizer, loss_fn=loss_fn, epochs=2, log_n_last_epochs=2)
```

# Demo

Check out the [Demo Notebook](https://nbviewer.jupyter.org/github/vshulyak/torch-es/blob/master/examples/torches_demo.ipynb).


# 📚 Dependencies

- torch
- numpy
