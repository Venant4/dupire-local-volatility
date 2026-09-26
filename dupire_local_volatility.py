import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import (
    griddata,
    SmoothBivariateSpline,
    RegularGridInterpolator
)
from scipy.stats import norm


# ============================================================
# 1. DATA PREPARATION
# ============================================================

DATA_PATH = "data/option.csv"

df = pd.read_csv(DATA_PATH)

# Select SPX European call options
df = df[df["underlying_symbol"] == "^SPX"]
df = df[df["option_type"] == "C"].copy()

# Convert dates
df["quote_datetime"] = pd.to_datetime(df["quote_datetime"])
df["expiration"] = pd.to_datetime(df["expiration"])

# Observation date
date_obs = pd.Timestamp("2023-09-21 16:15:00")

df = df[df["quote_datetime"] == date_obs].copy()

# Time to maturity in years
df["T"] = (
    (df["expiration"] - date_obs).dt.total_seconds()
    / (365 * 24 * 3600)
)

# Market price: bid-ask midpoint
df["market_price"] = (df["bid"] + df["ask"]) / 2

# Spot price
S0 = df["active_underlying_price"].median()

# Strike range
df = df[
    (df["strike"] >= 0.8 * S0)
    & (df["strike"] <= 1.2 * S0)
].copy()

# Remove unrealistic implied volatilities
df = df[
    (df["implied_volatility"] > 0.01)
    & (df["implied_volatility"] < 1.5)
].copy()


# ============================================================
# 2. IMPLIED VOLATILITY SURFACE
# ============================================================

K = df["strike"].values
T = df["T"].values
IV = df["implied_volatility"].values

# Regular grid
K_grid = np.linspace(K.min(), K.max(), 100)
T_grid = np.linspace(T.min(), T.max(), 100)

K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

# Interpolation
IV_mesh = griddata(
    (K, T),
    IV,
    (K_mesh, T_mesh),
    method="linear"
)

# Plot interpolated implied volatility surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    K_mesh,
    T_mesh,
    IV_mesh,
    cmap="viridis"
)

ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Implied Volatility")
ax.set_title("SPX Implied Volatility Surface")

plt.tight_layout()
plt.show()


# ============================================================
# 3. SMOOTHING OF THE VOLATILITY SURFACE
# ============================================================

mask = ~np.isnan(IV_mesh)

K_smooth = K_mesh[mask]
T_smooth = T_mesh[mask]
IV_smooth = IV_mesh[mask]

# Normalize variables for numerical stability
K_norm = K_smooth / S0

T_min = T_smooth.min()
T_max = T_smooth.max()

T_norm = (T_smooth - T_min) / (T_max - T_min)

# Bivariate spline
spline_IV = SmoothBivariateSpline(
    K_norm,
    T_norm,
    IV_smooth,
    s=0.5
)

K_mesh_norm = K_mesh / S0

T_mesh_norm = (
    T_mesh - T_min
) / (T_max - T_min)

# Smoothed volatility surface
IV_lisse = spline_IV.ev(
    K_mesh_norm.ravel(),
    T_mesh_norm.ravel()
).reshape(K_mesh.shape)


# Plot smoothed surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    K_mesh,
    T_mesh,
    IV_lisse,
    cmap="viridis"
)

ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Implied Volatility")
ax.set_title("Smoothed Implied Volatility Surface")

plt.tight_layout()
plt.show()


# ============================================================
# 4. BLACK-SCHOLES CALL
# ============================================================

def black_scholes_call(S, K, r, sigma, T):
    """
    Black-Scholes price of a European call option.
    """

    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


# Risk-free rate
r = 0.04


# ============================================================
# 5. CONVERT VOLATILITY SURFACE INTO PRICE SURFACE
# ============================================================

Price_surface = np.zeros_like(IV_lisse)

for i in range(IV_lisse.shape[0]):
    for j in range(IV_lisse.shape[1]):

        sigma = IV_lisse[i, j]
        maturity = T_mesh[i, j]
        strike = K_mesh[i, j]

        Price_surface[i, j] = black_scholes_call(
            S0,
            strike,
            r,
            sigma,
            maturity
        )


# ============================================================
# 6. NUMERICAL DERIVATIVES
# ============================================================

dK = K_grid[1] - K_grid[0]
dT = T_grid[1] - T_grid[0]

# First derivative with respect to maturity
dC_dT = np.gradient(
    Price_surface,
    dT,
    axis=0
)

# First derivative with respect to strike
dC_dK = np.gradient(
    Price_surface,
    dK,
    axis=1
)

# Second derivative with respect to strike
d2C_dK2 = np.gradient(
    dC_dK,
    dK,
    axis=1
)


# ============================================================
# 7. DUPIRE LOCAL VOLATILITY
# ============================================================

numerator = (
    dC_dT
    + r * K_mesh * dC_dK
    - r * Price_surface
)

denominator = (
    0.5
    * K_mesh**2
    * d2C_dK2
)

local_vol_squared = numerator / denominator

# Keep only valid points
valid_mask = (
    (d2C_dK2 > 0)
    & (local_vol_squared > 0)
    & np.isfinite(local_vol_squared)
)

local_vol = np.full_like(
    local_vol_squared,
    np.nan
)

local_vol[valid_mask] = np.sqrt(
    local_vol_squared[valid_mask]
)


# Plot local volatility surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    K_mesh,
    T_mesh,
    local_vol,
    cmap="viridis"
)

ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Local Volatility")
ax.set_title("Dupire Local Volatility Surface")

plt.tight_layout()
plt.show()


# ============================================================
# 8. LOCAL VOLATILITY INTERPOLATOR
# ============================================================

local_vol_clean = local_vol.copy()

# Replace missing values with the median
median_local_vol = np.nanmedian(local_vol_clean)

local_vol_clean[
    np.isnan(local_vol_clean)
] = median_local_vol

# Interpolator expects axes (T, K)
local_vol_interpolator = RegularGridInterpolator(
    (T_grid, K_grid),
    local_vol_clean,
    bounds_error=False,
    fill_value=None
)


# ============================================================
# 9. EULER-MARUYAMA SIMULATION
# ============================================================

def simulate_local_volatility(
    S0,
    r,
    T,
    N,
    steps,
    interpolator,
    K_min,
    K_max,
    T_min,
    T_max,
    seed=42
):
    """
    Simulate asset paths under the local volatility model
    using the Euler-Maruyama scheme.
    """

    dt = T / steps

    rng = np.random.default_rng(seed)

    Z = rng.normal(
        0,
        1,
        size=(N, steps)
    )

    paths = np.zeros(
        (N, steps + 1)
    )

    paths[:, 0] = S0

    for i in range(N):

        for j in range(steps):

            # Current time
            t = j * dt

            # Current spot
            S = paths[i, j]

            # Keep the spot inside the calibrated surface
            S_used = np.clip(
                S,
                K_min,
                K_max
            )

            # Keep time inside the calibrated surface
            t_used = np.clip(
                t,
                T_min,
                T_max
            )

            # Local volatility
            point = [[
                t_used,
                S_used
            ]]

            sigma_local = interpolator(
                point
            )[0]

            # Euler-Maruyama
            S_next = (
                S
                + r * S * dt
                + sigma_local
                * S
                * Z[i, j]
                * np.sqrt(dt)
            )

            # Avoid negative prices
            paths[i, j + 1] = max(
                S_next,
                0
            )

    return paths


# ============================================================
# 10. EUROPEAN CALL PRICING
# ============================================================

def price_european_call(
    S0,
    K,
    r,
    T,
    N,
    steps,
    interpolator
):
    """
    Monte Carlo price of a European call.
    """

    paths = simulate_local_volatility(
        S0,
        r,
        T,
        N,
        steps,
        interpolator,
        K_grid.min(),
        K_grid.max(),
        T_grid.min(),
        T_grid.max()
    )

    ST = paths[:, -1]

    payoff = np.maximum(
        ST - K,
        0
    )

    price = (
        np.exp(-r * T)
        * np.mean(payoff)
    )

    return price


# ============================================================
# 11. ASIAN CALL PRICING
# ============================================================

def price_asian_call(
    S0,
    K,
    r,
    T,
    N,
    steps,
    interpolator
):
    """
    Monte Carlo price of an arithmetic-average Asian call.
    """

    paths = simulate_local_volatility(
        S0,
        r,
        T,
        N,
        steps,
        interpolator,
        K_grid.min(),
        K_grid.max(),
        T_grid.min(),
        T_grid.max()
    )

    average_spot = np.mean(
        paths,
        axis=1
    )

    payoff = np.maximum(
        average_spot - K,
        0
    )

    price = (
        np.exp(-r * T)
        * np.mean(payoff)
    )

    return price


# ============================================================
# 12. UP-AND-OUT CALL PRICING
# ============================================================

def price_up_and_out_call(
    S0,
    K,
    r,
    T,
    N,
    steps,
    interpolator,
    barrier
):
    """
    Monte Carlo price of an Up-and-Out call.
    """

    paths = simulate_local_volatility(
        S0,
        r,
        T,
        N,
        steps,
        interpolator,
        K_grid.min(),
        K_grid.max(),
        T_grid.min(),
        T_grid.max()
    )

    ST = paths[:, -1]

    # Check whether the barrier was reached
    barrier_hit = (
        np.max(paths, axis=1)
        >= barrier
    )

    payoff = np.where(
        barrier_hit,
        0,
        np.maximum(ST - K, 0)
    )

    price = (
        np.exp(-r * T)
        * np.mean(payoff)
    )

    return price


# ============================================================
# 13. APPLICATION
# ============================================================

# Reference option
K_option = 4410
T_option = 0.077597

# Monte Carlo parameters
N = 10_000
steps = 100

# Barrier
H = 4800


# European Call
price_call = price_european_call(
    S0,
    K_option,
    r,
    T_option,
    N,
    steps,
    local_vol_interpolator
)


# Asian Call
price_asian = price_asian_call(
    S0,
    K_option,
    r,
    T_option,
    N,
    steps,
    local_vol_interpolator
)


# Up-and-Out Call
price_up_and_out = price_up_and_out_call(
    S0,
    K_option,
    r,
    T_option,
    N,
    steps,
    local_vol_interpolator,
    H
)


# ============================================================
# 14. RESULTS
# ============================================================

print("\n========== DUPIRE LOCAL VOLATILITY ==========")

print(f"Spot price: {S0:.4f}")
print(f"Risk-free rate: {r:.2%}")

print("\nLocal volatility statistics:")

print(
    f"Minimum: "
    f"{np.nanmin(local_vol):.4f}"
)

print(
    f"Maximum: "
    f"{np.nanmax(local_vol):.4f}"
)

print(
    f"Mean: "
    f"{np.nanmean(local_vol):.4f}"
)

print(
    f"Median: "
    f"{np.nanmedian(local_vol):.4f}"
)

print("\n========== MONTE CARLO PRICES ==========")

print(
    f"European Call: "
    f"{price_call:.4f}"
)

print(
    f"Asian Call: "
    f"{price_asian:.4f}"
)

print(
    f"Up-and-Out Call: "
    f"{price_up_and_out:.4f}"
)
