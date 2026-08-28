# An Analytic Brownian-Motion Model of a Galton Board

## 1 Physical picture

Consider a Galton board of height $H$. Beads are released with a uniform horizontal distribution over the opening

$-a \le x_{0} \le a$

while the side walls of the board are at $x = \pm b$, with $b \gg a$. Each collision with a peg produces a small, approximately random horizontal displacement. The vertical coordinate can therefore be treated as a time-like variable, and the repeated peg collisions can be modelled as one-dimensional diffusion in the horizontal direction.

The aim is to predict the standard deviation of the horizontal bead distribution at the bottom of the board.

## 2 Diffusion equation

Let $p(x, z)$ be the probability density for finding a bead at horizontal position $x$ after it has fallen a vertical distance $z$. The Brownian approximation gives

$\frac{\partial p}{\partial z} = D_{z} \frac{\partial^{2} p}{\partial x^{2}}$ (1)

where $D_{z}$ is an effective horizontal diffusion coefficient per unit vertical distance. Its dimensions are length, since $[D_{z}] = [x^{2}/z]$.

The uniform release condition is

$p(x, 0) = \begin{cases} \frac{1}{2a}, & |x| \le a, \\ 0, & |x| > a. \end{cases}$ (2)

The mean initial position is zero and the initial variance is

$\text{Var}(X_{0}) = \int_{-a}^{a} x^{2} \frac{dx}{2a} = \frac{a^{2}}{3}.$ (3)

## 3 Variance at the bottom of a wide board

If the side walls are sufficiently far away that almost no bead reaches them, the final horizontal position can be written as

$X(H) = X_{0} + \Delta X(H)$ (4)

where $X_{0}$ is the release position and $\Delta X(H)$ is the displacement produced by the peg collisions. Brownian diffusion gives

$\text{Var}[\Delta X(H)] = 2D_{z}H.$ (5)

The release position and subsequent Brownian displacement are independent. Their variances therefore add:

$\sigma^{2}(H) = \frac{a^{2}}{3} + 2D_{z}H.$ (6)

Thus the predicted standard deviation at the bottom is

$\sigma(H) = \sqrt{\frac{a^{2}}{3} + 2D_{z}H}.$ (7)

The result has two contributions. The term $a^{2}/3$ is the variance already present because the beads are released across a finite-width opening. The term $2D_{z}H$ is the additional variance accumulated through collisions with the pegs.

## 4 Connection with a discrete peg model

Suppose neighbouring peg rows are separated by a vertical distance $\Delta z$. If a collision at each row gives an independent horizontal displacement with zero mean and variance $s^{2}$ then after

$N = \frac{H}{\Delta z}$

rows, the collision-induced variance is $Ns^{2}$. Comparing this with $2D_{z}H$ gives

$D_{z} = \frac{s^{2}}{2\Delta z}$ (8)

Equation (7) can consequently be written as

$\sigma(H) = \sqrt{\frac{a^{2}}{3} + \frac{s^{2}H}{\Delta z}} = \sqrt{\frac{a^{2}}{3} + Ns^{2}}.$ (9)

For an ideal left-or-right step of magnitude $s$, the same expression follows directly from the variance of a symmetric random walk.

## 5 Shape of the final distribution

The final density is the convolution of the initial uniform distribution and the Gaussian Brownian propagator. In the absence of wall effects,

$p(x, H) = \frac{1}{4a} \left[ \text{erf}\left(\frac{x+a}{\sqrt{4D_{z}H}}\right) - \text{erf}\left(\frac{x-a}{\sqrt{4D_{z}H}}\right) \right]$ (10)

At small heights, this resembles a slightly smoothed uniform distribution. At large heights, when $2D_{z}H \gg a^{2}/3$, it becomes approximately Gaussian with standard deviation $\sqrt{2D_{z}H}$.

## 6 Role of the board walls

The wide-board result is valid only while the distribution remains well inside $[-b, b]$. A simple condition is

$\sqrt{\frac{a^{2}}{3} + 2D_{z}H} \ll b.$ (11)

A more conservative practical test is $3\sigma(H) < b$ which places the walls more than approximately three standard deviations from the centre.

If the side walls reflect the beads and the board is extremely tall, the distribution eventually approaches a uniform distribution on $[-b, b]$. In that limiting case,

$\sigma(H) \longrightarrow \frac{b}{\sqrt{3}}$ (12)

Thus diffusion predicts an initial growth according to Equation (7), followed eventually by saturation if wall collisions become common. When $b \gg a$ and Equation (11) is satisfied, $b$ does not appear in the leading prediction.

## 7 Using the model with data

For fixed $a$, Equation (6) predicts a linear relationship

$\sigma^{2} = \frac{a^{2}}{3} + 2D_{z}H$ (13)

Measurements of $\sigma^{2}$ for boards of different heights can therefore be plotted against $H$. The predicted intercept is $a^{2}/3$, and the gradient is $2D_{z}$. This provides a direct way to estimate the effective diffusion coefficient and to test whether the Brownian approximation is appropriate.

The model assumes independent, unbiased peg collisions with approximately constant step statistics. Systematic asymmetry, correlated collisions, bead-bead interactions, missing pegs, or position-dependent geometry can produce deviations from the predicted linear growth of $\sigma^{2}$.