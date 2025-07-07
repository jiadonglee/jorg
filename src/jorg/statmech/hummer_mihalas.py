"""
Hummer-Mihalas Occupation Probability Formalism
===============================================

This module implements the occupation probability formalism from Hummer and Mihalas 1988,
exactly following Korg.jl's implementation in statmech.jl (lines 346-661).

This provides corrections to hydrogen partition functions based on pressure ionization
and neutral/charged particle interactions in stellar atmospheres.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Optional

from ..constants import (
    bohr_radius_cgs, RydbergH_eV, eV_to_cgs, 
    electron_charge_cgs, kboltz_eV
)


@jit
def hummer_mihalas_w(T: float, n_eff: float, nH: float, nHe: float, ne: float, 
                    use_hubeny_generalization: bool = False) -> float:
    """
    Calculate the correction, w, to the occupation fraction of a hydrogen energy level 
    using the occupation probability formalism from Hummer and Mihalas 1988, optionally 
    with the generalization by Hubeny+ 1994.
    
    This is a direct translation of Korg.jl's hummer_mihalas_w function (lines 346-400).
    
    The expression for w is in equation 4.71 of H&M. K, the QM correction used is 
    defined in equation 4.24. Note that H&M's "N"s are numbers (not number densities), 
    and their "V" is volume. These quantities appear only in the form N/V, so we use 
    the number densities instead.
    
    This is based partially on Paul Barklem and Kjell Eriksson's WCALC fortran routine
    (part of HBOP.f), which is used by (at least) Turbospectrum and SME. As in that 
    routine, we do consider hydrogen and helium as the relevant neutral species, and 
    assume them to be in the ground state. All ions are assumed to have charge 1. 
    Unlike that routine, the generalization to the formalism from Hubeny+ 1994 is 
    turned off by default because I haven't closely checked it. The difference effects 
    the charged_term only, and temperature is only used when use_hubeny_generalization 
    is set to True.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    n_eff : float
        Effective principal quantum number of the hydrogen level
    nH : float  
        Number density of neutral hydrogen [cm^-3]
    nHe : float
        Number density of neutral helium [cm^-3] 
    ne : float
        Electron number density [cm^-3]
    use_hubeny_generalization : bool, optional
        Whether to use Hubeny+ 1994 generalization (default: False)
        
    Returns:
    --------
    float
        Occupation probability correction factor w
    """
    
    # Contribution to w from neutral species (neutral H and He, in this implementation)
    # This is sqrt<r^2> assuming l=0. I'm unclear why this is the approximation barklem uses.
    r_level = jnp.sqrt(5.0/2.0 * n_eff**4 + 1.0/2.0 * n_eff**2) * bohr_radius_cgs
    
    # How do I reproduce this helium radius?
    neutral_term = (nH * (r_level + jnp.sqrt(3.0) * bohr_radius_cgs)**3 + 
                   nHe * (r_level + 1.02 * bohr_radius_cgs)**3)
    
    # Contributions to w from ions (these are assumed to be all singly ionized, so n_ion = n_e)
    # K is a QM correction defined in H&M '88 equation 4.24
    K = jnp.where(
        n_eff > 3,
        # WCALC drops the final factor, which is nearly within 1% of unity for all n
        16.0/3.0 * (n_eff / (n_eff + 1.0))**2 * ((n_eff + 7.0/6.0) / (n_eff**2 + n_eff + 1.0/2.0)),
        1.0
    )
    
    # Binding energy
    chi = RydbergH_eV / n_eff**2 * eV_to_cgs
    e = electron_charge_cgs
    
    # Charged particle contribution
    charged_term = jnp.where(
        use_hubeny_generalization,
        # This is a straight line-by-line port from HBOP. Review and rewrite if used.
        jnp.where(
            (ne > 10) & (T > 10),
            # Hubeny generalization implementation
            _hubeny_charged_term(ne, T, K, n_eff),
            0.0
        ),
        # Standard H&M formulation
        16.0 * ((e**2) / (chi * jnp.sqrt(K)))**3 * ne
    )
    
    return jnp.exp(-4.0 * jnp.pi / 3.0 * (neutral_term + charged_term))


@jit
def _hubeny_charged_term(ne: float, T: float, K: float, n_eff: float) -> float:
    """
    Helper function for Hubeny+ 1994 generalization.
    
    This is a straight line-by-line port from HBOP. Review and rewrite if used.
    """
    A = 0.09 * jnp.exp(0.16667 * jnp.log(ne)) / jnp.sqrt(T)
    X = jnp.exp(3.15 * jnp.log(1 + A))
    BETAC = 8.3e14 * jnp.exp(-0.66667 * jnp.log(ne)) * K / n_eff**4
    F = 0.1402 * X * BETAC**3 / (1 + 0.1285 * X * BETAC * jnp.sqrt(BETAC))
    return jnp.log(F / (1 + F)) / (-4.0 * jnp.pi / 3.0)


def hummer_mihalas_U_H(T: float, nH: float, nHe: float, ne: float, 
                      use_hubeny_generalization: bool = False) -> float:
    """
    Calculate the partition function of neutral hydrogen using the occupation probability 
    formalism from Hummer and Mihalas 1988.
    
    This is a direct translation of Korg.jl's hummer_mihalas_U_H function (lines 435-661).
    
    Note: This is experimental, and not used by Korg for spectral synthesis.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    nH : float
        Number density of neutral hydrogen [cm^-3]
    nHe : float
        Number density of neutral helium [cm^-3]
    ne : float
        Electron number density [cm^-3]
    use_hubeny_generalization : bool, optional
        Whether to use Hubeny+ 1994 generalization (default: False)
        
    Returns:
    --------
    float
        Partition function of neutral hydrogen with occupation probability correction
    """
    
    # These are from NIST, but it would be nice to generate them on the fly.
    # Direct copy from Korg.jl lines 445-511
    hydrogen_energy_levels = jnp.array([
        0.0,
        10.19880615024,
        10.19881052514816,
        10.19885151459,
        12.0874936591,
        12.0874949611,
        12.0875070783,
        12.0875071004,
        12.0875115582,
        12.74853244632,
        12.74853299663,
        12.7485381084,
        12.74853811674,
        12.74853999753,
        12.748539998,
        12.7485409403,
        13.054498182,
        13.054498464,
        13.054501074,
        13.054501086,
        13.054502042,
        13.054502046336,
        13.054502526,
        13.054502529303,
        13.054502819633,
        13.22070146198,
        13.22070162532,
        13.22070313941,
        13.22070314214,
        13.220703699081,
        13.22070369934,
        13.220703978574,
        13.220703979103,
        13.220704146258,
        13.220704146589,
        13.220704258272,
        13.320916647,
        13.32091675,
        13.320917703,
        13.320917704,
        13.320918056,
        13.38596007869,
        13.38596014765,
        13.38596078636,
        13.38596078751,
        13.385961022639,
        13.4305536,
        13.430553648,
        13.430554096,
        13.430554098,
        13.430554262,
        13.462451058,
        13.462451094,
        13.46245141908,
        13.462451421,
        13.46245154007,
        13.486051554,
        13.486051581,
        13.486051825,
        13.486051827,
        13.486051916,
        13.504001658,
        13.504001678,
        13.50400186581,
        13.504001867,
        13.50400193582
    ])
    
    # Direct copy from Korg.jl lines 513-579
    hydrogen_energy_level_degeneracies = jnp.array([
        2,
        2,
        2,
        4,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        8,
        10,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        8,
        10,
        10,
        12,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6
    ])
    
    # Direct copy from Korg.jl lines 581-647
    hydrogen_energy_level_n = jnp.array([
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        9,
        10,
        10,
        10,
        10,
        10,
        11,
        11,
        11,
        11,
        11,
        12,
        12,
        12,
        12,
        12
    ])
    
    # For each level calculate the correction, w, and add the term to U
    # The expression for w comes from Hummer and Mihalas 1988 equation 4.71 
    U = 0.0
    for i, (E, g, n) in enumerate(zip(hydrogen_energy_levels, 
                                     hydrogen_energy_level_degeneracies,
                                     hydrogen_energy_level_n)):
        # Effective quantum number (times Z, which is 1 for hydrogen)
        n_eff = jnp.sqrt(RydbergH_eV / (RydbergH_eV - E))
        
        # Calculate occupation probability correction
        w = hummer_mihalas_w(T, n_eff, nH, nHe, ne, 
                           use_hubeny_generalization=use_hubeny_generalization)
        
        # Add contribution to partition function
        U += w * g * jnp.exp(-E / (kboltz_eV * T))
    
    return U


def validate_hummer_mihalas_implementation():
    """
    Validate the Hummer-Mihalas implementation against expected behavior.
    """
    
    print("\n=== VALIDATING HUMMER-MIHALAS IMPLEMENTATION ===")
    
    # Test conditions
    T = 5778.0  # Solar temperature
    nH = 1e15   # Typical hydrogen density in stellar photosphere
    nHe = 1e14  # Typical helium density  
    ne = 1e13   # Typical electron density
    
    print(f"Test conditions: T={T}K, nH={nH:.0e}, nHe={nHe:.0e}, ne={ne:.0e}")
    
    # Test w function for different n_eff values
    print(f"\nOccupation probability corrections (w):")
    n_eff_values = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    for n_eff in n_eff_values:
        w_standard = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False)
        w_hubeny = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=True)
        
        print(f"  n_eff={n_eff:4.1f}: w_standard={float(w_standard):.6f}, w_hubeny={float(w_hubeny):.6f}")
        
        # Basic sanity checks
        assert 0.0 <= float(w_standard) <= 1.0, f"w should be between 0 and 1, got {w_standard}"
        assert 0.0 <= float(w_hubeny) <= 1.0, f"w should be between 0 and 1, got {w_hubeny}"
    
    # Test partition function calculation
    print(f"\nHydrogen partition functions:")
    
    # Standard partition function (should be ~2 for low densities)
    U_standard = 2.0  # Standard H I partition function
    
    # With occupation probability correction
    U_hm_standard = hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False)
    U_hm_hubeny = hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=True)
    
    print(f"  Standard U_H: {U_standard:.3f}")
    print(f"  H&M standard: {float(U_hm_standard):.3f} (ratio: {float(U_hm_standard)/U_standard:.3f})")
    print(f"  H&M Hubeny:   {float(U_hm_hubeny):.3f} (ratio: {float(U_hm_hubeny)/U_standard:.3f})")
    
    # Basic sanity checks
    assert float(U_hm_standard) > 0, "Partition function should be positive"
    assert float(U_hm_hubeny) > 0, "Partition function should be positive"
    assert float(U_hm_standard) <= U_standard, "H&M correction should reduce partition function"
    
    # Test pressure dependence
    print(f"\nPressure dependence:")
    ne_values = [1e11, 1e12, 1e13, 1e14, 1e15]
    
    for ne_test in ne_values:
        U_test = hummer_mihalas_U_H(T, nH, nHe, ne_test, use_hubeny_generalization=False)
        ratio = float(U_test) / U_standard
        print(f"  ne={ne_test:.0e}: U={float(U_test):.3f} (ratio: {ratio:.3f})")
        
        # Higher pressure should reduce partition function more
        assert 0.0 < ratio <= 1.0, f"Partition function ratio should be between 0 and 1, got {ratio}"
    
    print(f"\n✅ All Hummer-Mihalas validation tests passed!")


if __name__ == "__main__":
    validate_hummer_mihalas_implementation()