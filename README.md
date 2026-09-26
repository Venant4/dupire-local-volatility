# Dupire Local Volatility Model

## Overview

Implementation of the Dupire local volatility model using SPX option data.

The project covers:

- Implied volatility surface calibration
- Smoothing and interpolation
- Local volatility computation using the Dupire equation
- Euler-Maruyama simulation
- Monte Carlo pricing of derivatives

## Mathematical Framework

The local volatility is obtained from:

\[
\sigma_{loc}^2(K,T)
=
\frac{
\frac{\partial C}{\partial T}
+rK\frac{\partial C}{\partial K}
-rC
}{
\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}
}
\]

The underlying is simulated under:

\[
dS_t=rS_tdt+\sigma_{loc}(S_t,t)S_tdW_t.
\]

## Applications

- European Call
- Asian Call
- Up-and-Out Call

## Technologies

Python · NumPy · Pandas · SciPy · Matplotlib

## Author

**Venant Nibaruta**  
Finance & Decision Engineering — ENSA Agadir
