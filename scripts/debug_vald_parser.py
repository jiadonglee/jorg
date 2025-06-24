#!/usr/bin/env python3
"""
Debug VALD parser to understand why it's not parsing lines
"""

import sys
from pathlib import Path

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent))

from jorg.lines.linelist import create_large_vald_linelist, parse_vald_line
import tempfile

def debug_vald_parsing():
    """Debug VALD parsing step by step"""
    
    print("🔍 VALD Parser Debug")
    print("=" * 40)
    
    # Create a simple test linelist
    simple_vald = """# VALD3 Extract Test
# Simple test case
#
'5889.9510',   0.108,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5895.9242',  -0.194,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'6562.8010',   0.640,  10.199,'H 1', 6.14e7, 2.8e-5, 1.4e-7, 0.3
"""
    
    print("Test VALD content:")
    print(simple_vald)
    
    # Test line-by-line parsing
    print("\n🧪 Testing individual line parsing:")
    
    test_lines = [
        "'5889.9510',   0.108,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30",
        "'5895.9242',  -0.194,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30",
        "'6562.8010',   0.640,  10.199,'H 1', 6.14e7, 2.8e-5, 1.4e-7, 0.3"
    ]
    
    for i, line in enumerate(test_lines):
        print(f"\nLine {i+1}: {line}")
        
        try:
            result = parse_vald_line(line, "auto", None)
            if result:
                print(f"  ✅ Parsed: λ={result.wavelength*1e8:.3f}Å, log_gf={result.log_gf:.3f}, species={result.species_id}")
            else:
                print(f"  ❌ Failed to parse")
                
                # Debug the parsing step by step
                line_text = line.replace("'", "").strip()
                print(f"    Cleaned: {line_text}")
                
                if ',' in line_text:
                    parts = [p.strip() for p in line_text.split(',')]
                else:
                    parts = line_text.split()
                
                parts = [p for p in parts if p]
                print(f"    Parts ({len(parts)}): {parts}")
                
                if len(parts) >= 4:
                    try:
                        wl = float(parts[0])
                        log_gf = float(parts[1])
                        E_lower = float(parts[2])
                        species_str = parts[3]
                        print(f"    Basic parsing: wl={wl}, gf={log_gf}, E={E_lower}, sp='{species_str}'")
                    except Exception as e:
                        print(f"    Basic parsing error: {e}")
                        
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    # Test full file parsing
    print(f"\n📖 Testing full file parsing:")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vald', delete=False) as f:
        f.write(simple_vald)
        test_file = f.name
    
    try:
        from jorg.lines import read_linelist
        
        linelist = read_linelist(test_file, format="vald")
        print(f"  Result: {len(linelist)} lines parsed")
        
        if len(linelist) > 0:
            for i, line in enumerate(linelist):
                print(f"    Line {i+1}: λ={line.wavelength*1e8:.3f}Å, log_gf={line.log_gf:.3f}")
        
    except Exception as e:
        print(f"  ❌ Full parsing error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        Path(test_file).unlink()

if __name__ == "__main__":
    debug_vald_parsing()