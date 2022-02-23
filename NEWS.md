# `pybats-detection` 0.0.3

  - Changed the names of model components. Now we are using the same name as `pybats`
  - Included the sum of seasonality components in the smooth and filter posterior moments
  - Fixed a index bug to compute the smooth distributions
  - Included the `verbose` argument in `Monitor` class, allowing the user to control the monitor detection
  - Allowed `None` values for the parameters of subbjective and noise type intervention, `h_shift`, `H_shift`, `a_star`, and `R_star` in `Intervention` class
  - Added a new quick start guide vignette
