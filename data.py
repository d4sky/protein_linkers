t2s = {}
t2s["GLY"] = 'G'
t2s["ALA"] = 'A'
t2s["VAL"] = 'V'
t2s["LEU"] = 'L'
t2s["ILE"] = 'I'

t2s["PHE"] = 'F'
t2s["PRO"] = 'P'
t2s["THR"] = 'T'
t2s["SER"] = 'S'
t2s["TYR"] = 'Y'

t2s["TRP"] = 'W'
t2s["ASP"] = 'D'
t2s["ASN"] = 'N'
t2s["GLU"] = 'E'
t2s["GLN"] = 'Q'

t2s["LYS"] = 'K'
t2s["ARG"] = 'R'
t2s["HIS"] = 'H'
t2s["MET"] = 'M'
t2s["CYS"] = 'C'

s2t = {aac:aan for aan, aac in t2s.items()}

# Protein Torsion Angle Sampling Lists (Phi, Psi)
# Level 1: Core clusters (7 samples)
ramachandran_level_1 = [
  (-119, 113),  # Parallel Beta-sheet
  (-139, 135),  # Anti-parallel Beta-sheet

  ( -70, 140),   # Polyproline II

  ( -57, -47),   # Alpha-helix (Right)
  ( -49, -26),   # 3-10 Helix
  
  (  60,  45),   # Alpha-helix (Left)
  (  80,  10)    # Gamma-turn / Type II turn. Originally (80,-80)
]

# Level 2: Extensive sampling (35 samples)
# Includes transitions and periphery of allowed regions
ramachandran_level_2 = [
  # --- Beta-Sheet / Extended Cluster ---
  (-119, 113),  # Parallel Beta-sheet
  (-127, 124),
  (-111, 133),
  (-104, 122),

  (-139, 135),  # Anti-parallel Beta-sheet
  (-126, 142),
  (-115, 150),
  (-140, 155),

  ( -70, 140),   # Polyproline II
  ( -77, 134),
  ( -58, 136),
  ( -74, 151),

  ( -57, -47),   # Alpha-helix (Right)
  ( -62, -39),
  ( -64, -44),
  ( -67, -37),
  
  ( -49, -26),   # 3-10 Helix
  ( -72, -15),   # 51
  ( -82, -22),   # 52
  ( -88,  -7),   # 53
  ( -98,   5),   # 54
  
  (  60,  45),   # Alpha-helix (Left)
  (  80,  10),   # Gamma-turn / Type II turn. Originally (80,-80)
  (  56,  39),
  (  63,  30),
  (  71,  21),
]