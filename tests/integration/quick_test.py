#!/usr/bin/env python3
"""Quick test of molecular implementation"""

import sys
from pathlib import Path

# Add Jorg to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from jorg.lines.molecular_species import get_molecular_species, STELLAR_MOLECULES
    
    # Print all available molecules
    print("Available molecules:")
    for name in STELLAR_MOLECULES.keys():
        print(f"  {name}")
    
    # Test TiO
    print("\nTesting TiO...")
    print(f"Direct lookup: {STELLAR_MOLECULES.get('TiO')}")
    print(f"Upper case lookup: {STELLAR_MOLECULES.get('TiO'.upper())}")
    
    tio = get_molecular_species('TiO')
    print(f"TiO result: {tio}")
    
    # Try upper case
    tio_upper = get_molecular_species('TIO')
    print(f"TIO result: {tio_upper}")
    
    if tio:
        print(f"TiO name: {tio.name}")
        print(f"TiO ID: {tio.species_id}")
        print(f"TiO mass: {tio.mass_amu}")
    
    # Test other molecules
    molecules = ['H2O', 'OH', 'CO']
    for mol in molecules:
        species = get_molecular_species(mol)
        if species:
            print(f"✅ {mol}: Found")
        else:
            print(f"❌ {mol}: Not found")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()