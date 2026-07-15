# PINN Atmosphere

This repository contains a simple physics-informed neural network (PINN) test
case for the one-dimensional inviscid Euler equations. The current problem is a
smooth periodic entropy wave with an analytical solution. It is meant as a clean
starting point before moving to more complex atmospheric-flow equations.

## Problem

The PINN learns primitive variables

```text
rho(x,t), u(x,t), p(x,t)
```

while enforcing the conservative 1D Euler equations:

```text
q_t + f(q)_x = 0
```

The analytical entropy-wave solution is used to generate training and validation
data.

## Folder Structure

```text
PINN_Atmosphere/
├── data/
├── train/
├── train_plots/
├── predict_plots/
├── equations_for_check.py
├── requirements.txt
├── best_model.pt
└── final_model.pt
```

`best_model.pt` and `final_model.pt` are created after training. They are ignored
by git because they are generated model checkpoints.

## data/

This folder contains the analytical solution and generated datasets.

```text
data/
├── __init__.py
├── exact_solution.py
├── generate_data.py
└── generated/
```

`data/exact_solution.py`

Defines the analytical entropy-wave solution:

```text
rho = 1 + epsilon*sin(2*pi*(x - u0*t))
u   = u0
p   = p0
```

It also provides the conservative variables for the exact solution.

`data/generate_data.py`

Generates `.npz` files used for training and validation.

`data/generated/`

Contains generated datasets:

```text
initial_condition.npz
boundary_condition.npz
collocation.npz
sensor_data.npz
validation_grid.npz
```

## train/

This folder contains the PINN model, physics residuals, condition losses, and
training driver.

```text
train/
├── __init__.py
├── model.py
├── train.py
├── data_io.py
├── physics/
│   └── euler_1d.py
├── conditions/
│   ├── initial.py
│   ├── boundary.py
│   └── sensors.py
└── losses/
    └── assembly.py
```

`train/model.py`

Defines `EulerPINN`, the neural network. It maps `(x,t)` to primitive variables
`rho`, `u`, and `p`. Density and pressure are passed through `softplus` so they
remain positive.

`train/train.py`

Main training script. It loads generated data, creates the model, computes the
PINN loss, validates on the held-out validation grid, and saves:

```text
best_model.pt
final_model.pt
```

in the repository root.

`train/data_io.py`

Loads `.npz` datasets and converts them to PyTorch tensors.

`train/physics/euler_1d.py`

Contains the 1D Euler physics:

```text
primitive_to_conservative()
euler_fluxes()
gradient()
euler_residuals()
```

`train/conditions/initial.py`

Computes the initial-condition loss.

`train/conditions/boundary.py`

Computes the periodic boundary-condition loss.

`train/conditions/sensors.py`

Computes the sparse sensor-data loss.

`train/losses/assembly.py`

Combines the physics, initial-condition, boundary-condition, and sensor-data
losses into one weighted training loss.

## train_plots/

Contains plotting code and generated training-loss figures.

```text
train_plots/
└── plot_training.py
```

`train_plots/plot_training.py`

Reads `best_model.pt`, extracts the saved loss history, and generates:

```text
training_losses_linear.png
training_losses_log.png
validation_loss.png
```

## predict_plots/

Contains plotting code and generated prediction-error figures.

```text
predict_plots/
└── plot_prediction.py
```

`predict_plots/plot_prediction.py`

Loads `best_model.pt`, evaluates the trained PINN on
`data/generated/validation_grid.npz`, compares against the exact validation
solution, and generates:

```text
rho_profiles.png
u_profiles.png
p_profiles.png
rho_error_heatmap.png
u_error_heatmap.png
p_error_heatmap.png
```

## equations_for_check.py

Reference implementation of the Euler residual equations. This file is kept as a
diagnostic/reference copy separate from the modular training loss code.

## requirements.txt

Python dependencies:

```text
torch
numpy
matplotlib
```

## How to Run

From the repository root:

```bash
cd /Users/saviopoovathingal/Downloads/PINN_Atmosphere
```

Train the model:

```bash
.venv/bin/python train/train.py
```

Run a shorter training test:

```bash
.venv/bin/python train/train.py --epochs 100 --log-every 10
```

Generate training-loss plots:

```bash
.venv/bin/python train_plots/plot_training.py
```

Generate prediction/validation plots:

```bash
.venv/bin/python predict_plots/plot_prediction.py
```

## Outputs

Training creates:

```text
best_model.pt
final_model.pt
```

Training plots are saved in:

```text
train_plots/
```

Prediction comparison plots are saved in:

```text
predict_plots/
```

## Extending Beyond 1D Euler

The current code is organized so the simple 1D Euler problem can be replaced by
more complex governing equations later. For a new PINN problem, the main areas
to change are the physics equations, neural-network outputs, data files, and
loss assembly.

### 1. Physics Equations

Current file:

```text
train/physics/euler_1d.py
```

This file defines the PDE residuals used in the physics loss. For a more complex
model, add a new physics file instead of overwriting the Euler example:

```text
train/physics/wrf_flux_form.py
train/physics/navier_stokes_2d.py
train/physics/euler_3d.py
```

The new file should provide a residual function with the same general role as
`euler_residuals()`: evaluate the model, compute derivatives with autograd, and
return residual tensors that should be driven toward zero.

For atmospheric flow, this is where WRF-style flux-form equations would go.

### 2. Neural-Network Outputs

Current file:

```text
train/model.py
```

The current model predicts:

```text
rho, u, p
```

For a larger system, update the model outputs. For example, an atmospheric model
might predict:

```text
U, V, W, Theta_m, mu_d, phi, Q_m
```

or another set of primitive variables that are later converted into conservative
or diagnostic variables.

### 3. Data Files

Current folder:

```text
data/generated/
```

The current `.npz` files contain arrays such as:

```text
x, t, rho, u, p
```

A new problem will need data files with the coordinates and variables required
by the new model and loss terms. For example:

```text
x, y, eta, t, U, V, W, Theta_m, mu_d, phi, Q_m
```

The loader in `train/data_io.py` can stay mostly the same if the data is still
stored as `.npz` arrays.

### 4. Conditions and Loss Terms

Current folder:

```text
train/conditions/
```

This contains losses for:

```text
initial condition
periodic boundary condition
sensor data
```

For a more complex problem, add or replace condition files, for example:

```text
train/conditions/inflow.py
train/conditions/wall.py
train/conditions/top_boundary.py
train/conditions/observations.py
```

Then update:

```text
train/losses/assembly.py
```

to combine the new physics and condition losses.

### Recommended Pattern

Do not overwrite the simple Euler example immediately. Add new modules beside it:

```text
train/physics/euler_1d.py
train/physics/wrf_flux_form.py
```

Then choose which physics module to use inside `train/losses/assembly.py`.

This keeps the simple problem available as a working test case while adding more
realistic atmospheric-flow PINNs.
