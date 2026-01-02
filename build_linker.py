import sys, os, re
import numpy as np

from support import Ensemble, Sphere
from curves import Curve

def find_linker(inp_structure_path, cChain, nChain, nRes = 15, level = 1, incr = 0.1, colThr = 0.60):
  tmol = Ensemble(inp_structure_path)
  if any(v is None for v in tmol.idxs[cChain].iloc[-1]['bb_idxs']):
    print(f"Last residue in the growing chain '{cChain}' is not fully defined. Terminating...")    
    return None
    
  last_resi  = tmol.get_last_resi(cChain)
  first_resi = tmol.get_first_resi(nChain)
  
  print(f"Last Residue in the Chain '{cChain}': {last_resi}" )
  print(f"First Residue in the Chain '{nChain}': {first_resi}")
  
  p1 = tmol.get_coors(f"{cChain}/{last_resi}/C")[0]
  p2 = tmol.get_coors(f"{nChain}/{first_resi}/N")[0]
  
  tLine = Curve(100)
  tLine.create_from_line(p1, p2)
  tLine.fit_spline()  
  tLine.Write_pdb("line.pdb")
  
  score, _ = tmol.check_curve_collision(tLine)
  if score < 5.0:
    tmol.make_linker(cChain, tLine)
    return tmol
  else:
    print("Straigh line is not usable. Going to scan anothe possibilities")
    for perimeter in np.linspace(incr, 1.0, int(1.0/incr)):
      print(f"Perimeter:, {perimeter:.3f}")  
      span = np.linalg.norm(p2-p1)/2.0
      ellipsoid = Sphere(span*perimeter, span*perimeter, span, 50, 10)
      ellipsoid.align_to_points(p2, p1)
      results = []
      for mi, meridian in enumerate(ellipsoid.meridians):
        tCurve = Curve()
        tCurve.create_from_points(meridian)
        #tCurve.Write_pdb(f"meridian_{mi}.pdb")
        #score, _ = tmol.check_curve_collision(tCurve)
        score = tmol.interaction_energy_with_curve(tCurve)
        if score < colThr:
          results.append((mi, score, tCurve))
      if len(results) > 0:
        print("Found plausible Curve")
        results.sort(key=lambda x: x[1])
        for mi, score, tCurve in results[:1]:
          print(f"Direct Length: {tCurve.length:.3f}")
          tCurve.Write_pdb(f"selected_poludnik.pdb")
          tmol.make_linker(cChain, tCurve, level, nRes)
          return tmol
      else:
        print("No plausible Curve found. Trying next with expanding path")
  return None      
  
'''
structure_file_path = "4g0n.pdb"
#structure_file_path = "2I4Q.pdb"
#structure_file_path = "2I4Q_e.pdb"

#chainC, chainN = 'A', 'B'
chainC, chainN = 'B', 'A'
'''

structure_file_path = "5XJR.pdb"
#chainC, chainN = '2', 'B'
chainC, chainN = 'B', '2'

nResis = 25
out_file = structure_file_path.replace(".pdb", f"_{chainC}_{chainN}_linker.pdb")

mol_with_liner = find_linker(structure_file_path, chainC, chainN, nResis, level = 2, colThr = 1.20)
if mol_with_liner:
  mol_with_liner.Write_pdb(out_file)
