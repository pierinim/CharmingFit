# CharmingFit — UTfit-style Bayesian analysis for charmless two-body B decays

<p align="center">
  <img src="logo/logo_white.png" width="750" alt="CharmingFit Logo">
</p>

This package performs a Bayesian (Monte Carlo importance-sampling) fit of
charmless two-body decays of B+, B0, and Bs mesons into ππ, Kπ, and KK final states.

- **Amplitudes:** SU(2) isospin basis, with SU(3)-breaking factors in polar form.
- **CKM inputs:** Flat priors on (λ, A, ρ̄, η̄) within UTfit 2022 ranges.
- **Observables:** Branching ratios, direct A_CP, and S coefficients from CP analyses.
- **Output:** ROOT ntuple with sampled parameters and likelihood weights.

Main entry point:
```bash
python run_fit.py --observables data/observables.yaml --out charmless_fit.root -N 500000
