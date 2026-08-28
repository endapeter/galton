# Brownian-Motion Framing of the Galton Board: Research Plan

*Created 2026-08-22. Formalizes goals 5-6 of `PLAN.md` (step 9 of the experiment queue):
describe the board's input → output distribution transform in the language of Brownian
motion / diffusion processes — first as pure mathematics, then as the forward process of
a neural generative model.*

---

## 1. The central hypothesis

Let vertical descent play the role of time (`t` ∝ number of peg rows crossed) and let
`X_t` be the horizontal coordinate. The hypothesis to formalize and test:

> **The board approximates a heat semigroup acting on the input density:**
> `p_out ≈ (e^{t·L} p_in)`, where `L` is a Fokker–Planck generator with
> (approximately) constant diffusion coefficient `D` and no drift.

Existing evidence (steps 7-8, `figures/`):

- Step 7: for narrow funnels σ_out saturates at the board's *intrinsic* mixing
  σ_board ≈ 11.19 — the signature of a smoothing operator with a fixed kernel width.
- Step 8: Gaussian input, σ_in = 15.63 → σ_out = 19.68. **Quadrature check:** if the
  board is pure additive diffusion, σ_out = √(σ_in² + σ_board²) = √(15.63² + 11.19²)
  = **19.22** vs. observed **19.68** (2.4% error). The heat-kernel hypothesis is
  already quantitatively consistent.

---

## 2. Pure-mathematics program

### 2.1 Invariance principle with correlated increments

Naively, `X_n` is a sum of peg-impact kicks and Donsker's invariance principle gives
weak convergence of the rescaled walk to Brownian motion, predicting
`X_t ~ N(0, 2Dt)`. But the kicks are **not i.i.d.**: the trajectory is a deterministic
dynamical system (a gravity-loaded periodic Lorentz gas) with randomness only in the
initial conditions. The correct theorem templates:

- **Chernov & Lebowitz**, *The Galton board: limit theorems and recursion*,
  J. Stat. Phys. **86**, 895-914 (1997) — rigorous CLT for exactly our model class:
  periodic scatterer lattice, gravity, restitution < 1.
- **Bunimovich & Sinai** (1981), diffusion in the field-free periodic Lorentz gas —
  the archetype of "deterministic → diffusive".
- **Kipnis & Varadhan**, Commun. Math. Phys. **104**, 1-19 (1986) — CLT for additive
  functionals of Markov chains; gives the correct variance constant when increments
  are correlated (the correction to `E[(Δx)²] = 2Dt`).
- **Meyn & Tweedie**, *Markov Chains and Stochastic Stability* (1993) — ergodicity
  tools for the one-row chain below.

**Task (M1):** estimate `D` from the existing per-ball CSVs via `E[(Δx)²] = 2Dt`,
predict `p_out = N(0, 2Dt)`, and localize where it breaks. Known breakpoints from
step 6: peg radius ≈ gap ((2.0, 0.5) and (2.5, 0.6) give zero throughput; R² = 0.65
at (3.0, 0.7)) — the diffusion limit should fail exactly there.

### 2.2 The board as a Markov kernel and a Fokker–Planck PDE

- Define the **one-row transition kernel** `K(x, x')` empirically (a ball entering
  row `n` at horizontal position `x` exits at `x'`). The whole board is `K^50`
  (50 rows), so `p_out = K^50 p_in`. Study the spectrum of `K`: if `K` ≈ heat kernel,
  eigenvalues decay like `e^{-λ_k t}` and powers of `K` stay in the same family.
- Continuum limit — the **Fokker–Planck equation**:
  `∂_t p = −∂_x(b(x)p) + ½ ∂²_x(σ²(x)p)`,
  with reflecting boundary conditions at the walls (Risken, *The Fokker–Planck
  Equation*, 1984). Position-dependent `b, σ` encode funnel geometry and peg-density
  modulation; a density gradient of pegs is a *drift* term — the design lever for
  distribution shaping (goal 2 of `problem.txt`).
- **Homogenization**: the fast spatial periodicity of the peg lattice should average
  to an effective constant `D` predictable from lattice geometry + restitution alone
  (Bensoussan, Lions & Papanicolaou, *Asymptotic Analysis for Periodic Structures*,
  1978). A closed-form `D(d, r, e)` would replace every parameter sweep.
- Wall confinement + long boards → convergence to a **stationary distribution**
  (eigenfunction expansion of the Fokker–Planck operator; compare
  Ornstein–Uhlenbeck). Testable by sweeping board length `N_ROWS`.

### 2.3 CLT-violation targets

- **Heavy-tailed flight times → α-stable limits** via the generalized CLT
  (Gnedenko & Kolmogorov). The Lorentz-gas precedent: **infinite-horizon**
  superdiffusion, where `E[X_t²] ~ t log t` instead of `t` (Bleher, J. Stat. Phys.
  **66**, 315-343, 1992). Concrete mechanism to hunt for: **open corridors in the
  peg lattice** (remove pegs to create unbounded free paths) — a design recipe, not
  just an observation.
- **Arcsine law** (goal 3): Lévy's arcsine law for the fraction of time a Brownian
  path spends positive; target an output density with arcsine shape (bimodal at the
  walls) via trapping/reflection regimes. Mechanism to explain the "gaps": the
  stationary mass concentrates near attractors of the one-row kernel.

### 2.4 Vibration → time-dependent coefficients

An oscillating board makes drift and diffusion **periodic in time**: the
Fokker–Planck operator has periodic coefficients, and Floquet analysis of that
operator gives the averaged output distribution. Resonance (a `notes.txt` question)
= parametric resonance: vibration frequency matching the mean peg-hit frequency
amplifies `σ(t)` multiplicatively. Diagnostics needed: mean free path / mean free
time between peg hits (already listed in `notes.txt`).

### 2.5 Beyond-Markov correction

Low restitution creates memory between kicks. If increments have long-range
correlation, the phenomenological replacement is **fractional Brownian motion**
(Kolmogorov 1940; Mandelbrot & Van Ness 1968) with Hurst index `H ≠ ½`. Test by
variance scaling: `E[X_t²] ~ t^{2H}` — a straight log-log diagnostic on trajectory
data; `H > ½` would already be a CLT violation of the persistence type.

---

## 3. Neural-network program

The inversion: **the board is a physical forward ("noising") process — precisely the
object diffusion and score-based generative models learn to reverse.** The repo's GPU
engine (one kernel launch = 2000 forward samples) is the forward simulator; the CSVs
already contain the `(initial_x, final_x)` couplings that training needs.

### 3.1 The board as the forward SDE of a score-based model

In score-based SDE terms (Song et al., ICLR 2021), the board implements a
variance-exploding-type forward process `dX = √(2D) dW` (pure diffusion, no drift),
so `X_t = X_0 + √(2Dt)·ε`. Then:

- The **reverse SDE** `dX = [−2D ∇_x log p_t(X)]dt + √(2D) dW̄` recovers the input
  distribution from outputs — *un-Galton a distribution* — once a network
  `s_θ(x, t) ≈ ∇_x log p_t(x)` is trained on forward samples.
- Falsifiable prediction: if the board is exactly a heat kernel, the exact reverse is
  the backward heat equation (ill-posed); the network must regularize. The learned
  denoiser should satisfy Tweedie's formula `E[X_0 | X_t] = X_t + t·∇log p_t` — an
  internal consistency check on the trained model.
- Discrete-time counterpart: DDPM (Ho, Jain & Abbeel, NeurIPS 2020) with **50
  non-learned, physics-defined diffusion steps** (one per peg row).

### 3.2 Learning the transform directly

- **Neural SDEs** (Kidger, Foster, Li & Lyons, NeurIPS 2021): fit drift and diffusion
  of an SDE to trajectory data by gradient descent → the continuum Fokker–Planck
  model of §2.2, learned rather than postulated. Cross-check: learned `D` must match
  the estimator `E[(Δx)²] = 2Dt` from §2.1.
- **Conditional flow matching** (Lipman et al., ICLR 2023; Tong et al., NeurIPS
  2023): the input→output coupling is a noisy transport; CFM regresses a velocity
  field from output back to input, giving a learnable inverse transport that stays
  on the data manifold by construction (sidesteps backward-heat ill-posedness).
- **Schrödinger bridge** (De Bortoli et al., NeurIPS 2021): the maximum-entropy
  stochastic process connecting the *measured* input and output marginals — a
  principled "effective board" model computable from unpaired samples at both ends
  (exactly what the step-7 sweep provides: `initial_x`, `final_x` per ball).

### 3.3 PDE-constrained learning (ties the two programs together)

Postulate the Fokker–Planck PDE of §2.2 and fit `b(x), σ(x)` with a physics-informed
network (Raissi, Perdikaris & Karniadakis, J. Comput. Phys. **378**, 686-707, 2019;
Sirignano & Spiliopoulos, J. Comput. Phys. **375**, 1339-1367, 2018): minimize PDE
residual + data-matching loss. The learned coefficients and the NN experiments of
§3.1-3.2 must agree with the estimator-based `D` of §2.1 — three independent roads
to the same constant, a strong falsification design.

### 3.4 Concrete NN experiments (priority order)

1. Train a score/denoiser on the `(initial_x, final_x)` pairs already in
   `figures/step7_sigma_vs_funnel/` and `figures/step8_gaussian_input/`; evaluate the
   recovered input distribution against held-out inputs (Wasserstein-2 / MMD).
2. Sweep σ_in (extends step 8) to obtain a family of `(p_in, p_out)` pairs; fit a
   conditional flow; test whether **one** learned inverse works across the sweep —
   i.e. whether the board transform is a fixed linear smoothing operator (the
   heat-kernel hypothesis again, now tested in function space).
3. Neural SDE on trajectories; compare the learned diffusion coefficient with `D`
   from §2.1.

---

## 4. Milestones

| # | Deliverable | Needs |
|---|---|---|
| M1 | `D` estimate + Gaussian validation vs. `N(0, 2Dt)`; breakpoint map | existing CSVs only — no new sims |
| M2 | Empirical one-row kernel `K`; spectrum; `p_out = K^50` vs. observed | small instrumented runs |
| M3 | Score-based inverse ("un-Galton") experiment | existing step-7/8 CSVs + training loop |
| M4 | CLT-violation design study: opened-lattice superdiffusion (`E[X²] ~ t log t`), arcsine target | geometry variants |
| M5 | Vibration: periodic Fokker–Planck / Floquet analysis + resonance criterion | moving-peg engine (architecture change) |

M1 is the gate: if the heat-kernel prediction matches everywhere except the known
breakpoints, §3's generative inversion is well-posed as a regularized deconvolution
and M3 proceeds; where it fails, §2.3's anomalous-diffusion program takes over.

---

## 5. References

**Stochastic-process foundations**

- A. Einstein, *Über die von der molekularkinetischen Theorie der Wärme geforderte
  Bewegung…*, Ann. Phys. 17, 549 (1905) — the diffusion–Brownian-motion link.
- M. D. Donsker, *An invariance principle for certain probability limit theorems*
  (1951) — random walk → Brownian motion.
- H. Risken, *The Fokker–Planck Equation: Methods of Solution and Applications*
  (Springer, 1984).
- I. Karatzas & S. E. Shreve, *Brownian Motion and Stochastic Calculus* (1991) —
  reflecting boundaries, heat-kernel / semigroup formalism.
- A. N. Kolmogorov (1940); B. B. Mandelbrot & J. W. Van Ness, SIAM Rev. 10, 422
  (1968) — fractional Brownian motion.
- B. V. Gnedenko & A. N. Kolmogorov, *Limit Distributions for Sums of Independent
  Random Variables* — generalized (stable) CLT.

**Deterministic diffusion / Galton board**

- L. Bunimovich & Ya. G. Sinai, Commun. Math. Phys. 78, 479 (1981) — diffusion in
  the periodic Lorentz gas.
- N. Chernov & J. L. Lebowitz, *The Galton board: limit theorems and recursion*,
  J. Stat. Phys. 86, 895-914 (1997) — **the** rigorous CLT for the gravity-loaded
  board with restitution.
- C. Kipnis & S. R. S. Varadhan, Commun. Math. Phys. 104, 1 (1986) — CLT for
  correlated (Markov-chain) increments.
- P. M. Bleher, J. Stat. Phys. 66, 315 (1992) — infinite-horizon Lorentz gas,
  `t log t` superdiffusion — the CLT-violation mechanism.
- S. P. Meyn & R. L. Tweedie, *Markov Chains and Stochastic Stability* (1993).
- A. Bensoussan, J.-L. Lions & G. Papanicolaou, *Asymptotic Analysis for Periodic
  Structures* (North-Holland, 1978) — homogenization → effective `D`.
- S. Redner, *A Guide to First-Passage Processes* (Cambridge, 2001) — hitting-time
  diagnostics for the vibration/resonance question.

**Neural networks / generative modeling**

- J. Ho, A. Jain & P. Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS
  (2020).
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon & B. Poole,
  *Score-Based Generative Modeling through Stochastic Differential Equations*,
  ICLR (2021).
- V. De Bortoli, J. Thornton, J. Heng & A. Doucet, *Diffusion Schrödinger Bridge
  with Applications to Score-Based Generative Modeling*, NeurIPS (2021).
- M. Lipman, Y. Li, H. Nicklin, Y. Zhang & R. T. Q. Chen, *Flow Matching for
  Generative Modeling*, ICLR (2023); A. Tong et al., *Improving and generalizing
  flow-based generative models with minibatch optimal transport*, NeurIPS (2023).
- P. Kidger, J. Foster, X. Li & T. J. Lyons, *Neural SDEs as Infinite-Dimensional
  GANs*, ICML (2021).
- M. Raissi, P. Perdikaris & G. E. Karniadakis, *Physics-informed neural networks*,
  J. Comput. Phys. 378, 686 (2019); J. Sirignano & K. Spiliopoulos, *DGM: a deep
  learning algorithm for solving partial differential equations*, J. Comput. Phys.
  375, 1339 (2018).
