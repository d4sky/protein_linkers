import os, sys
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from data import *

current_module_path = os.path.abspath(__file__) # Get the absolute path 
base_dir = os.path.dirname(current_module_path) # Get the directory containing this file 

lib_dir  = "libraries"
res_dir  = "libraries/aa_building_blocks/"

INF    = 1e10
Adim   = 180
PI     = np.pi
PIFAC  = 180/PI
BETA_3 = 4.730040744862817

patchMol = None
res_lib = {} # Dictionary to store read Amino Acids to avoid re-reading them

def is_integer(inp):
  try:
    int(inp)
    return True
  except ValueError:
    return False
    
def is_float(inp):
  try:
    float(inp)
    return True
  except ValueError:
    return False

def Cout(lines, file_name):
  with open(file_name, 'w') as f:
    f.write('\n'.join(lines))
    print(f"Wrote file: {file_name}")

def Get_dihedral_value(u1, u2, u3):
  '''
  u12   = u1.Crossp(u2)
  u23   = u2.Crossp(u3)

  v_r   = u12.Crossp(u23)
  f_a   = u12.Dotp(u23)
  angle = math.atan2(u2.Dotp(v_r),u2.Abs()*f_a)            # Rotational angle
  '''
  
  u12 = np.cross(u1, u2)
  u23 = np.cross(u2, u3)
  
  v_r = np.cross(u12, u23)
  f_a = np.dot(u12, u23)
    
  angle = np.arctan2(np.dot(u2, v_r), np.linalg.norm(u2) * f_a)

  return angle
  
def kabsch(P, Q):
    # P and Q are 3x3 matrices: columns are the triangle points in system B and A respectively
    # P, B - is moving system
    # Q, A - is static system
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:  # reflection correction
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    t = centroid_Q - R @ centroid_P

    return R, t

def angle_dist(a, b):
  d = abs(a - b) % 360
  return d if d <= 180 else 360 - d
    
def get_psis_from_ramachandra_plot(phi, tol = 20.0, level = 1):
  if level == 1:
    ramachandran_plot = ramachandran_level_1
  else:
    ramachandran_plot = ramachandran_level_2
      
  selected_psis = [
    pair[1] for pair in ramachandran_plot
    if angle_dist(pair[0], phi) <= tol
  ]
  
  if not selected_psis:
    # Fallback: find the single closest phi in the data
    closest_pair = min(ramachandran_plot, key=lambda p: angle_dist(p[0], phi))
    return [closest_pair[1]]
  
  return sorted(list(set(selected_psis)))
  
class Body():
  def __init__(self):
    self.center     = np.array([0.0, 0.0, 0.0])
    self.quaternion = np.array([1.0, 0.0, 0.0, 0.0]) # Orientation: [w, x, y, z] - Default identity (no rotation)
    
    # Physical properties
    self.mass     = 1.0
    self.inertia  = np.eye(3)  # 3x3 Identity matrix
    self.inerinv  = np.eye(3)
    
    self.evalues  = np.ones(3)
    self.evectors = np.eye(3)

  def calculate_physical_properties(self, coords, masses=None):
    """Calculates center of mass and inertia tensor from points."""
    if masses is None:
      masses = np.ones(len(coords))
    
    self.mass = np.sum(masses)
    # Center of mass
    #self.center = np.sum(coords * masses[:, np.newaxis], axis=0) / self.mass
    
    # Shift coordinates to center for inertia calculation
    # !!!! Caution !!!!
    #shifted_coords = coords - self.center
    shifted_coords = coords 
    
    # Inertia Tensor components
    x, y, z = shifted_coords[:, 0], shifted_coords[:, 1], shifted_coords[:, 2]
    
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * (x * y))
    Ixz = -np.sum(masses * (x * z))
    Iyz = -np.sum(masses * (y * z))
    
    self.inertia = np.array([
      [Ixx, Ixy, Ixz],
      [Ixy, Iyy, Iyz],
      [Ixz, Iyz, Izz]
    ])
    
  def shift(self, R_shift):
    self.center = self.center + R_shift    

class Shape(Body):
  def __init__(self, set_color='red'):
    super().__init__()
    # Visual/Geometry data
    self.color = set_color
    self.polygon_type = ''
    self.N_vertices = 0
    self.R = 1.0

    self.vertices = []  # These should ideally stay as "local" coordinates
    self.normals  = []
    self.colors   = []
    self.edges    = []
    self.polygons = []
    self.pcenters = []
    self.pnormals = []
    
    self.minR     = np.zeros(3)
    self.maxR     = np.zeros(3)

class Sphere(Shape):
  def __init__(self, a=1.0, b=None, c=None, Nf=20, Nt=40, set_color='red'):
    super().__init__(set_color)
    # Handle Ellipsoid parameters
    self.a = a
    self.b = b if b is not None else a
    self.c = c if c is not None else a
    self.R = a # Keeping R as primary radius
    
    self.polygon_type = 'triangles'
    phi_angles   = np.linspace(0, np.pi, Nf + 1)
    theta_angles = np.linspace(0, 2 * np.pi, Nt + 1)
    # 1. Generate unique vertex coordinates
    unique_vertices = []
    for i in range(Nf + 1):
      phi = phi_angles[i]
      for j in range(Nt + 1):
        theta = theta_angles[j]
        x = self.a * np.sin(phi) * np.cos(theta)
        y = self.b * np.sin(phi) * np.sin(theta)
        z = self.c * np.cos(phi)
        unique_vertices.append(np.array([x, y, z]))

    self.vertices = np.array(unique_vertices)
    self.N_vertices = len(self.vertices)
    # Helper to get the index in the 1D vertices list from 2D grid coordinates
    def get_idx(phi_idx, theta_idx):
      return phi_idx * (Nt + 1) + theta_idx
        
    self.polygons = []
    self.pcenters = []
    self.pnormals = []
    self.normals  = []
    edge_set = set()
    for i in range(Nf):
      for j in range(Nt):
        # Get indices of the 4 corners of the quad
        idx1 = get_idx(i, j)
        idx2 = get_idx(i + 1, j)
        idx3 = get_idx(i + 1, j + 1)
        idx4 = get_idx(i, j + 1)

        # Helper to process a triangle using indices
        def add_triangle_idx(i1, i2, i3):
          self.polygons.append([i1, i2, i3])
          v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
          # Face properties
          self.pcenters.append((v1 + v2 + v3) / 3.0)
          normal = np.cross(v2 - v1, v3 - v1)
          mag = np.linalg.norm(normal)
          self.pnormals.append(normal / mag if mag > 0 else np.array([0, 0, 1]))
          # Add unique edges as index pairs [idx_a, idx_b]
          for edge in [(i1, i2), (i2, i3), (i3, i1)]:
            edge_set.add(tuple(sorted(edge)))

        # Triangle 1
        if i != (Nf - 1): add_triangle_idx(idx1, idx2, idx3)
        # Triangle 2
        if i != 0: add_triangle_idx(idx1, idx3, idx4)

    # 2. Finalize edges as a list of index lists
    self.edges = [list(e) for e in edge_set]

    # 3. Calculate Vertex Normals (one per unique vertex)
    for i, v in enumerate(self.vertices):
      vn = np.array([v[0]/self.a**2, v[1]/self.b**2, v[2]/self.c**2])
      norm = np.linalg.norm(vn)
      self.normals.append(vn / norm if norm > 0 else np.array([0,0,1]))
    
    self.meridians = [] 
    for j in range(Nt + 1):
      meridian_coords = []
      for i in range(Nf + 1):
        idx = get_idx(i, j)
        meridian_coords.append(self.vertices[idx])
      self.meridians.append(np.array(meridian_coords))
      
    self.meridians = [] # Poludníky (Pole to Pole)
    for j in range(Nt + 1):
        self.meridians.append(self.vertices[j::(Nt + 1)])

    self.parallels = [] # Rovnobežky (Circles around the center)
    for i in range(Nf + 1):
        start = i * (Nt + 1)
        end = (i + 1) * (Nt + 1)
        self.parallels.append(self.vertices[start:end])
    
    # Update bounding box
    self.minR = np.array([-self.a, -self.b, -self.c])
    self.maxR = np.array([ self.a,  self.b,  self.c])

    # Calculate physical properties based on generated vertices
    # Assuming uniform mass for the shell/surface
    self.calculate_physical_properties(np.array(self.vertices))

  def calculate_physical_properties(self, masses=None):
    """
    Overrides Shape/Body method to use analytical formulas 
    for a solid ellipsoid instead of discrete points.
    """
    # Principal moments of inertia
    Ixx = (1/5) * self.mass * (self.b**2 + self.c**2)
    Iyy = (1/5) * self.mass * (self.a**2 + self.c**2)
    Izz = (1/5) * self.mass * (self.a**2 + self.b**2)
    
    # Off-diagonals are zero for an axis-aligned ellipsoid
    self.inertia  = np.array([
        [Ixx, 0.0, 0.0],
        [0.0, Iyy, 0.0],
        [0.0, 0.0, Izz]
    ])
    
    # Also pre-calculate the inverse for physics solvers
    self.inerinv  = np.linalg.inv(self.inertia)
    
    # Set eigenvalues/vectors (already aligned with identity)
    self.evalues  = np.array([Ixx, Iyy, Izz])
    self.evectors = np.eye(3)

  def get_rotation_between_vectors(self, a, b):
    """Helper to find rotation matrix that maps vector a to vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    
    if s < 1e-9: # Vectors are already aligned or opposite
        return np.eye(3) if c > 0 else -np.eye(3)
        
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Rodrigues' rotation formula
    R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s**2))
    return R

  def align_to_points(self, p1, p2):
    """
    Positions and orients the ellipsoid such that its 'a' axis 
    lies on the line segment connecting p1 and p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # 1. Calculate Center
    new_center = (p1 + p2) / 2.0
    
    # 2. Calculate Direction Vector
    vec_r = p2 - p1
    dist = np.linalg.norm(vec_r)
    if dist < 1e-9:
        return # Points are the same, no orientation possible
    
    target_dir = vec_r / dist  # Normalized direction
    
    # 3. Find Rotation from Local X [1,0,0] to target_dir
    #local_x = np.array([1.0, 0.0, 0.0])
    local_z = np.array([0.0, 0.0, 1.0])
    
    # Using a standard rotation matrix between two vectors
    # (Similar to your rmatrix_unit_vectors function)
    #rot_matrix = self.get_rotation_between_vectors(local_x, target_dir)
    rot_matrix = self.get_rotation_between_vectors(local_z, target_dir)
    
    if hasattr(self, 'center') and np.any(self.center != 0):
        self.vertices -= self.center
        self.pcenters -= self.center
    
    # 4. Apply Transformation to Geometry
    # Rotate first (around local origin), then shift to the midpoint
    self.vertices = self.vertices @ rot_matrix.T
    self.pcenters = self.pcenters @ rot_matrix.T
    self.pnormals = self.pnormals @ rot_matrix.T
    self.normals  = self.normals @ rot_matrix.T
    #self.normals = np.array(self.normals) @ rot_matrix.T
    
    # Shift to final world position
    self.center = new_center
    self.vertices += self.center
    self.pcenters += self.center
    
    for i in range(len(self.meridians)):
      self.meridians[i] = self.meridians[i] @ rot_matrix.T
      self.meridians[i] += self.center
      
    for i in range(len(self.parallels)):
      self.parallels[i] = self.parallels[i] @ rot_matrix.T
      self.parallels[i] += self.center

def parse_specs(inp_string):
  sel_ints_set = set()
  sel_strs = []
    
  items = [i.strip() for i in inp_string.split(',') if i.strip()]
  for item in items:
    if '-' in item:
      teils = item.split('-')
      if len(teils) == 2:
        # Check if both sides are integers
        if is_integer(teils[0]) and is_integer(teils[1]):
          beg, end = int(teils[0]), int(teils[1])
          if end >= beg:
            # Use .update() to add range to the existing set
            sel_ints_set.update(range(beg, end + 1))
        else:
          # If it has a '-' but isn't an int range (e.g. 'PE-1')
          if item not in sel_strs:
            sel_strs.append(item)
    else:
      if is_integer(item):
        sel_ints_set.add(int(item))
      else:
        if item not in sel_strs:
          sel_strs.append(item)
                
    return sorted(list(sel_ints_set)), sel_strs

    '''
    INFINITY = 1000000  
    sel_ints_set = set()
    sel_strs = []    
    for item in items:
      teils = item.split('-')
      if len(teils) == 2:
        beg, end = INFINITY, -INFINITY
        if is_integer(teils[0]) and is_integer(teils[1]): 
          beg = int(teils[0])
          end = int(teils[1])
        if end >= beg:
          sel_ints = [si for si in range(beg, end + 1) if si not in sel_ints]
      else:
        if is_integer(item):
          si = int(item)
          if si not in sel_ints: sel_ints.append(si)
        else:
          if len(item)>0:
            if item not in sel_strs: sel_strs.append(item.strip())
            
    return sel_ints, sel_strs
    '''
    
class Ensamble(Body):
  def __init__(self, file_name=None, chain_to_add_patches = False, verbose = False) -> object:
    super().__init__()
    # Descriptor table: one row per atom
    # Columns: chain, res_name, res_num, atom_name, element, occupancy, b_factor, etc.
    self.atoms = pd.DataFrame(
      columns=[ 'record',   # "ATOM" or "HETATM"
        'atom_id', 'atom_name', 'res_name',  'chain_id',
        'res_num',  'element',   'occupancy', 'b_factor'
      ]
    )
    self.coors  = None # Coordinates: Nx3 numpy array
    self.max    = np.zeros(3)
    self.min    = np.zeros(3)
    self.dim    = np.zeros(3)
    self.center = np.zeros(3)
    
    self.idxs  = {} #
    
    self.title = ""
    self.method = ""
    self.resolution = None

    self.active_chain = ''  # Chain to which/or which will be added. Nt,Ct,NP,CP will be calculated only for this chain
    self.Nt  = np.array([]) # First 3 atoms at N-terminus for ading new protein/amino acid 
    self.Ct  = np.array([]) # Last  3 atoms at C-terminus

    self.NP  = np.array([]) # Patch atoms at N-terminus to which new protein/amino acid will be added
    self.CP  = np.array([]) # Patch atoms at C-terminus
    
    if patchMol is not None:
      self.patchMol = patchMol
    else:
      self.patchMol = None
      
    self.pps    = {}
    self.models = {}    # Dictionary of coordinates for each mode - MODEL ID is the Key
    
    # Auto-load if file name is given
    if file_name:
      self.Read_pdb(file_name, chain_to_add_patches=chain_to_add_patches, verbose=verbose)
                
  def Set_active_chain(self, chain):
    if chain and chain in self.idxs:  
      self.active_chain = chain
      self.Set_termini()      
    else:
      if not chain:
        print(f"Chain to be set as active has to be defined")  
      else:
        print(f"Chain '{chain}' to be set as active is not in the Ensamble")  

  def Set_termini(self):
    sub_idxs = self.idxs[self.active_chain] #.dropna(subset=['res_num'])
    if sub_idxs.empty:
      nt_list, ct_list = None, None
    else:
      idx_min = sub_idxs['res_num'].idxmin()
      nt_list = sub_idxs.at[idx_min, 'bb_idxs']  
      idx_max = sub_idxs['res_num'].idxmax()
      ct_list = sub_idxs.at[idx_max, 'bb_idxs']  

    self.Nt, self.Ct  = np.asarray(nt_list, dtype=int), np.asarray(ct_list, dtype=int)
    self.Apply_patches()
   
  def Select(self, selection_string):
    '''
    RETURN the Selected indeces
    Without syntax Check
    inp_string has to be in format:
    chain1, chain2, ... chainN/resi_spec1, resi_spec2, .... resi_specN/atom_spec1, atom_spec2, ... atom_specN
    resi_spec = residue_number|residue_number_interval|residue_name
    atom_spec = atom_number|atom_number_interval|atom_name
    '''
    if selection_string == "all" or selection_string == "//":
      return self.atoms.index.to_numpy()

    try:
      slash_items = selection_string.split('/')
      if len(slash_items)!=3:
        print(f"Selection string '{selection_string}' could not be parsed. Taken as 'all'")
        return self.atoms.index.to_numpy()
        
      mask = pd.Series(True, index=self.atoms.index)        
      # 1. Chain Filter
      if slash_items[0]:
        sel_chains = [c.strip() for c in slash_items[0].split(',') if c.strip()]
        mask &= self.atoms['chain_id'].isin(sel_chains)          
      # 2. Residue Filter (Numbers and Names)
      if slash_items[1]:
        sel_resis, sel_resns = parse_specs(slash_items[1])
        if sel_resis and sel_resns:
          mask &= (self.atoms['res_num'].isin(sel_resis)) | (self.atoms['res_name'].isin(sel_resns))
        elif sel_resis:
          mask &= self.atoms['res_num'].isin(sel_resis)
        elif sel_resns:
          mask &= self.atoms['res_name'].isin(sel_resns)

      # 3. Atom Filter (IDs and Names)
      if slash_items[2]:
        sel_atmis, sel_atmns = parse_specs(slash_items[2])
        if sel_atmis and sel_atmns:
          mask &= (self.atoms['atom_id'].isin(sel_atmis)) | (self.atoms['atom_name'].isin(sel_atmns))
        elif sel_atmis:
          mask &= self.atoms['atom_id'].isin(sel_atmis)
        elif sel_atmns:
          mask &= self.atoms['atom_name'].isin(sel_atmns)          
            
      return self.atoms[mask].index.to_numpy()

    except Exception as e:
        print(f"Selection Error: {e}")
        return self.atoms.index.to_numpy()

  def shift(self, shiftV = np.zeros(3)):
    self.coors = self.coors + shiftV
    super().shift(shiftV)
    self.calculate_geometrical_properties()

  def calculate_geometrical_properties(self):
    self.min    = np.min(self.coors, axis=0)
    self.max    = np.max(self.coors, axis=0)
    self.dim    = self.max - self.min
    self.center = self.coors.mean(axis=0)
    
  def rotate(self, rotQ):
    rm = Rm_from_quaternion(rotQ)
    centered_verts = self.coors - self.center

    # Rotate the centered vertices
    rotated_centered_verts = np.dot(centered_verts, rm.T)
    # Shift them back to the current center
    self.coors = rotated_centered_verts + self.center
    #self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    self.calculate_geometrical_properties()
       
  def Edit_chain(self, selection_string='//', target_chain_id='A'):
    selected_atom_indices = self.Select(selection_string)
    self.atoms.loc[selected_atom_indices, 'chain_id'] = target_chain_id
    
    ''' # if iteration will be needed     
    for i in selected_atom_indices:
      # .at is faster than .loc for single-cell access
      self.atoms.at[i, 'chain_id'] = target_chain_id
    '''

  def Apply_patches(self, chain = ''): 
    if chain:
        pass
    else:        
        Nter_indexes = self.Nt[0:3]  # self N1,Ca1,C1
        Cter_indexes = self.Ct[-3:]  # self last Ca,C,O
        
    if self.patchMol is not None:      
        #Q_stat = self.coors[self.Nt[0:3]]                     # self N1,Ca1,C1
        Q_stat = self.coors[Nter_indexes]                      # self N1,Ca1,C1
        P_move = self.patchMol.coors[self.patchMol.Ct[-4:-1]]  # patch last N,Ca,C (skip its O if appropriate)
        R, t = kabsch(P_move, Q_stat)
    
        new_points = (R @ self.patchMol.coors.T).T + t
        #for_NPatch = [new_points[2], new_points[3], self.coors[self.Nt[0]]]   # Traingle O0 (Oxygen of zero's amino acid), C0, N1
        for_NPatch = [new_points[2], new_points[3], self.coors[Nter_indexes[0]]]   # Traingle O0 (Oxygen of zero's amino acid), C0, N1
        self.NP = np.array(for_NPatch)
        
        #-------------------------------------------------------------------------------------------------------------
        #Q_stat = self.coors[self.Ct[-3:]]                     # self last Ca,C,O
        Q_stat = self.coors[Cter_indexes]                      # self last Ca,C,O
        P_move = self.patchMol.coors[self.patchMol.Nt[1:4]]    # Patsch's 2nd, 3rd and fourth atoms, i.e. Ca, C, O of the patch residue
        R, t = kabsch(P_move, Q_stat)
    
        new_points = (R @ self.patchMol.coors.T).T + t
        #for_CPatch = [self.coors[self.Ct[-2]], self.coors[self.Ct[-1]], new_points[4]] 
        for_CPatch = [self.coors[self.Ct[-2]], self.coors[Cter_indexes[-1]], new_points[4]] 
        self.CP = np.array(for_CPatch)
    
  def Add_another_protein(self, other, direction="NC"):
    def _shift_tuple(tup, offset):
      if tup is None:
        return None
      arr = np.asarray(tup, dtype=int)
      return tuple((arr + offset).tolist())
  
    if other.NP is None or len(other.NP) == 0 or self.CP is None or len(self.CP) == 0:
      print("!!! Patches are not defined. Nothing done !!!")
      return 
      
    if direction == "NC":
      R, t = kabsch(other.NP, self.CP)
    else:  # "CN"
      R, t = kabsch(other.CP, self.NP)
    
    other.coors = (R @ other.coors.T).T + t
    other.NP    = (R @ other.NP.T).T + t
    other.CP    = (R @ other.CP.T).T + t
    
    if self.active_chain and other.active_chain:
      chain_id = self.active_chain  
      if direction == "NC":
        # Maximal Residue Number  
        if chain_id in self.idxs and not self.idxs[chain_id].empty:
          start_res = int(self.idxs[chain_id]['res_num'].max())
        else:
          # if we haven't built self.idxs[chain_id] (e.g. first time), find max from atoms
          mask_self = (self.atoms['chain_id'] == chain_id)
          start_res = int(self.atoms.loc[mask_self, 'res_num'].dropna().max()) if mask_self.any() else 0

        other_atoms = other.atoms[other.atoms['chain_id'] == other.active_chain].copy()
        other_atoms['chain_id'] = self.active_chain
        
        #mask = pd.Series(True, index=other_atoms.index) # All these atoms are now part of the merge
        #blocks = other_atoms.loc[mask, ['res_num']].drop_duplicates(ignore_index=True)
        blocks = other_atoms[['res_num']].dropna().drop_duplicates(ignore_index=True)
        
        new_nums = np.arange(int(start_res) + 1, int(start_res) + len(blocks) + 1, dtype=int)
        mapdict = dict(zip(blocks['res_num'], new_nums))
        
        #other_atoms.loc[mask, 'res_num'] = other_atoms.loc[mask, 'res_num'].map(mapdict).astype(int)
        other_atoms['res_num'] = other_atoms['res_num'].map(mapdict).astype(int)
        
        self.atoms = pd.concat([self.atoms, other_atoms], ignore_index=True)
        #self.coors = np.vstack([self.coors, other.coors])
        other_mask_indices = other.atoms.index[other.atoms['chain_id'] == other.active_chain].tolist()
        other_selected_coors = other.coors[other_mask_indices]
        self.coors = np.vstack([self.coors, other_selected_coors])
        
        #old_n_atoms = 0 if self.coors is None else (len(self.coors) - len(other.coors))  # coords length before vstack
        # if you kept a separate "old_n_atoms" before stacking, use that instead
        old_n_atoms = len(self.coors) - len(other_selected_coors)

        # Take the other chain’s residue index DF
        #other_idx_df = other.idxs.get(chain_id, pd.DataFrame()).copy()
        other_idx_df = other.idxs.get(other.active_chain, pd.DataFrame()).copy()
        other_idx_df['chain_id'] = self.active_chain

        # Shift atom indices by how many atoms we had before appending
        other_idx_df['bb_idxs'] = other_idx_df['bb_idxs'].apply(lambda t: _shift_tuple(t, old_n_atoms))
        other_idx_df['sc_idxs'] = other_idx_df['sc_idxs'].apply(lambda t: _shift_tuple(t, old_n_atoms))

        # Build a stable mapping for other residues in file order
        res_blocks = other_idx_df['res_num'].drop_duplicates().tolist()
        mapdict = {old: start_res + i + 1 for i, old in enumerate(res_blocks)}
        other_idx_df['res_num'] = other_idx_df['res_num'].map(mapdict).astype(int)
        
        # Append to our chain DF
        if chain_id in self.idxs and not self.idxs[chain_id].empty:
          self.idxs[chain_id] = pd.concat([self.idxs[chain_id], other_idx_df], ignore_index=True)
        else:
          self.idxs[chain_id] = other_idx_df
      else:
        print('!'*66)  
        print("FUNCTIONALITY TO ADD PROTEIN AT N-TERMINUS HAS TO BE IMPLEMENTED YET")
        sys.exit()
    else:
      print('!'*66)  
      print("ChainID(s) is/are missing")
      sys.exit()

    #self.pps = {} 
    #self.Calculate_phi_psi()
    self.calculate_geometrical_properties()
    self.Set_termini()

  def Add_residue(self, resn, chain = '', smer = "NC"):
    if resn not in res_lib:
      res_file_path = os.path.join(base_dir, res_dir, f"{resn}.pdb")
      res_mol = Ensamble(res_file_path, chain_to_add_patches=True)
      if chain:
        pass  
          
      res_lib[resn] = res_mol
    
    addin_res = res_lib[resn]
    self.Add_another_protein(addin_res, smer)

  def make_linker(self, aChain, target_curve, level = 1, maxIter = 15, eps = 0.05):
    self.Set_active_chain(aChain)
    t_min = 0.0
    for ai in range(maxIter):
      self.Add_residue("ALA")
      self.Calculate_phi_psi()
      resi = self.get_last_resi(aChain)
      print("Adding Resi", resi)
      res_idx  = self.get_res_idx(aChain, resi-1)
      prev_phi = self.pps[aChain][res_idx][0]
      psis     = get_psis_from_ramachandra_plot(prev_phi, level = level)
      if level == 1:
        phis = sorted(list(set(pair[0] for pair in ramachandran_level_1)))  # For more exhausing search
      else:
        phis = sorted(list(set(pair[0] for pair in ramachandran_level_2)))  # For more exhausing search
        
      res_idx  = self.get_res_idx(aChain, resi-1)
      rem_psi  = self.pps[aChain][res_idx][1]
      res_idx  = self.get_res_idx(aChain, resi)
      rem_phi  = self.pps[aChain][res_idx][0]
      c = 0
      rem_dist = 1e10
      for psi in psis:
        self.Set_torsion_angle(aChain, resi-1, psi, "psi")
        for phi in phis:
          self.Set_torsion_angle(aChain, resi, phi, "phi")
          c += 1
          #self.Write_pdb(f"z_{c}.pdb")
          checkCoor = self.get_coors(f"{aChain}/{resi}/C")
          #checkCoor = self.get_coors(f"{aChain}/{resi}/O")
          if t_min == 0.0:
            min_dist, t_min = target_curve.find_closest_distance(checkCoor)
          else:
            min_dist, t_min = target_curve.find_closest_limited_distance(checkCoor, t_min)
          #min_dist = np.linalg.norm(checkCoor - p2)
          if min_dist < rem_dist:
            rem_dist = min_dist
            rem_psi  = psi
            rem_phi  = phi
          if t_min >= 1.0 - eps:
            print("Reached the end of the Curve")
            self.Set_torsion_angle(aChain, resi-1, rem_psi, "psi")
            self.Set_torsion_angle(aChain, resi,   rem_phi, "phi")
            self.Calculate_phi_psi() 
            self.Set_termini()      
            self.calculate_geometrical_properties() 
            return 
      self.Set_torsion_angle(aChain, resi-1, rem_psi, "psi")
      self.Set_torsion_angle(aChain, resi,   rem_phi, "phi")
      self.Calculate_phi_psi() 
      self.Set_termini()      
      self.calculate_geometrical_properties() 
    print(f"Exceeded maximal number of iterations {maxIter}")
    return

  def get_coors(self, selection_string='//'):
    selected_atom_indices = self.Select(selection_string)
    return self.coors[selected_atom_indices]
    
  def Center(self):
    self.calculate_geometrical_properties()
    self.coors = self.coors - self.center  
    
  def calculate_physical_properties(self, verbose = False):
    self.calculate_geometrical_properties()
    r  = self.coors - self.center  # (N,3) displacements
    r2 = np.sum(r * r, axis=1)        # (N,) squared lengths
    I = np.eye(3)
    # Sum_i [ (|r_i|^2) I - r_i r_i^T ]
    self.inertia = np.sum(r2[:, None, None] * I - r[:, :, None] * r[:, None, :], axis=0)

    self.evalues, self.evectors = np.linalg.eigh(self.inertia)
    if verbose:
      print(self.evalues)
      print("Eigen Vectors")
      print(self.evectors)

  def Calculate_phi_psi(self):
    self.pps = {}
    for chain_id, sub in self.idxs.items():
      pps = []  
      N  = CA = C = np.zeros(3)
      nres = len(sub)
      for i in range(nres):
        if nres == 0:
          continue
          
        row = sub.iloc[i]
        #bb_idx = np.asarray(row.bb_idxs, dtype=int)  # [N, CA, C, O]
        bb_idx = row.bb_idxs  # [N, CA, C, O]
        
        # Cm (C of previous residue) or from patch NP
        if i == 0:
          if isinstance(self.NP, np.ndarray) and self.NP.size >= 2*3:
            Cm = self.NP[1]  # your previous convention
          else:
            pps.append((np.nan, np.nan)); continue
        else:
          Cm = C

        # Np (N of next residue) or from patch CP
        if i == nres - 1:
          if isinstance(self.CP, np.ndarray) and self.CP.size >= 3*3:
            Np = self.CP[2]
          else:
            pps.append((np.nan, np.nan)); continue
        else:
          #Np = self.coors[self.bb_idx[4*(i+1) + 0]]
          #bb_next = np.asarray(sub.iloc[i+1].bb_idxs, dtype=int)
          bb_next = sub.iloc[i+1].bb_idxs
          Np = self.coors[bb_next[0]]  # next residue's N

        # backbone atom coordinates for residue i
        N  = self.coors[bb_idx[0]]
        CA = self.coors[bb_idx[1]]
        C  = self.coors[bb_idx[2]]

        u1 = N  - Cm
        u2 = CA - N
        u3 = C  - CA
        u4 = Np - C

        phi = Get_dihedral_value(u1, u2, u3)
        psi = Get_dihedral_value(u2, u3, u4)

        pps.append((float(phi) * PIFAC, float(psi) * PIFAC))    
      # Finally asigning the pps for each CHAIN
      self.pps[chain_id] = pps

  def Print_phi_psi(self):
    for chain_id, pps in self.pps.items():
      print(chain_id)
      for pp in pps:
        print(pp)
         
  def Rotate_bb(self, chain_id, resi, angle_deg, what):
    df = self.idxs[chain_id]
    res_row = df.iloc[resi]
    bb_idx = tuple(res_row['bb_idxs'])
    sc_idx = tuple(res_row['sc_idxs'])

    angle_rad = np.deg2rad(angle_deg)
    if   what == "phi":
      axis_p1_gi = bb_idx[0]   # N_i
      axis_p2_gi = bb_idx[1]   # CA_i
      bb_to_move = bb_idx[1:]  # CA_i, C_i, O_i
      sc_to_move = sc_idx      # side chain of residue i rotates for phi
    elif what == "psi":
      # Axis: Ca_i (P1) -> C_i (P2)
      axis_p1_gi = bb_idx[1]   # CA_i
      axis_p2_gi = bb_idx[2]   # C_i
      bb_to_move = bb_idx[2:]  #C_i, O_i
      sc_to_move = tuple()     # side chain of residue i does NOT rotate for psi
    else:
      return

    downstream_bb = sum(df['bb_idxs'].iloc[resi+1:].tolist(), ())
    downstream_sc = sum(df['sc_idxs'].iloc[resi+1:].tolist(), ())
    bb_to_rotate = bb_to_move + downstream_bb
    sc_to_rotate = sc_to_move + downstream_sc
    
    P1 = self.coors[axis_p1_gi]
    P2 = self.coors[axis_p2_gi]
    axis = P2 - P1
    nrm = np.linalg.norm(axis)
    if nrm == 0.0:
        return
    axis = axis / nrm

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    all_to_rotate = np.array(bb_to_rotate + sc_to_rotate, dtype=int)
    
    pts = self.coors[all_to_rotate]
    shifted = pts - P1
    rotated = (R @ shifted.T).T + P1
    self.coors[all_to_rotate] = rotated

  def get_first_resi(self, chain):
    if chain in self.idxs:
      last_res_idx = self.idxs[chain].index[0]
      try:
        last_resi = self.idxs[chain].loc[last_res_idx]['res_num']
        return last_resi
      except:
        print(f"No atoms found for the chain '{chain}'")
        return None
    else:
      print(f"Chain '{chain}' not found in Ensamble")
      return None      

  def get_last_resi(self, chain):
    if chain in self.idxs:
      last_res_idx = self.idxs[chain].index[-1]
      try:
        last_resi = self.idxs[chain].loc[last_res_idx]['res_num']
        return last_resi
      except:
        print(f"No atoms found for the chain '{chain}'")
        return None
    else:
      print(f"Chain '{chain}' not found in Ensamble")
      return None      

  def get_res_idx(self, chain, resi):
    if chain in self.idxs:
      sub_idxs = self.idxs[chain]
      try:
        res_msk = sub_idxs['res_num']==resi
        return sub_idxs.index[res_msk][0]
      except:
        print(f"Reside '{resi}' not found within the chain '{chain}'")
        return None
    else:
      print(f"Chain '{chain}' not found in Ensamble")
      return None
    
  def Set_torsion_angle(self, chain, resi, angle, which = "phi"):
    if chain not in self.pps:
        self.Calculate_phi_psi()
        
    res_idx = self.get_res_idx(chain, resi)
    if res_idx is not None:  
      if chain in self.pps:
        if res_idx >= 0 and res_idx <= len(self.pps[chain]):
          phi, psi = self.pps[chain][res_idx]
          if which == "phi":
            self.Rotate_bb(chain, res_idx, angle - phi, "phi")
            #self.Rotate_bb(chain, res_idx, - phi, "phi")
            self.pps[chain][res_idx] = angle, psi
          else:
            self.Rotate_bb(chain, res_idx, angle - psi, "psi")
            #self.Rotate_bb(chain, res_idx, - psi, "psi")
            self.pps[chain][res_idx] = phi, angle
        else:
          print(f"No Torsion Angles Phi/Psi were found for the residue with index '{res_idx}' in chain '{chain}' ")
      else:
        print(f"No Torsion Angles Phi/Psi were found for the chain '{chain}'")

  def Set_phi_psi(self, chain_id, resi, phi_angle, psi_angle):
    self.Set_torsion_angle(chain_id, resi, phi_angle, "phi")
    self.Set_torsion_angle(chain_id, resi, psi_angle, "psi")
    
  def Set_phi_psi_angles(self, chain_id, resi_angles):
    self.Calculate_phi_psi()
    for res_num, (phi_t, psi_t) in resi_angles.items():
        self.Set_phi_psi(chain_id, res_num, phi_t, psi_t)

  def Orient(self, axis = 'x', principal='max'):
    self.Center()  
    self.calculate_physical_properties()  
    order = np.argsort(self.evalues)
    
    if principal == 'min':
        k = order[0]
    elif principal == 'mid':
        k = order[1]
    else:
        k = order[2]  # 'max' (default)

    self.R_p = self.evectors  # columns = principal axes in world coords
    if np.linalg.det(self.R_p) < 0:
        self.R_p[:, 0] *= -1.0  # flip one axis

    self.coors = self.coors @ self.R_p

    # 4) In the principal frame, e_k is the axis of interest. Aim it to target.
    e = np.eye(3)
    e_k = e[:, k]  # unit vector along chosen principal axis in principal frame
    target = {'x': e[:, 0], 'y': e[:, 1], 'z': e[:, 2]}[axis.lower()]
    self.U = rmatrix_unit_vectors(e_k, target)

    # Apply the in-frame alignment
    self.coors = self.coors @ self.U

  def check_curve_collision(self, curve, vdw_radius=1.7):
    """
    Calculates collision score without a grid by measuring 
    exact atom-to-curve distances.
    """
    #curve_pts = np.array(curve.points).T  # Shape: (num_samples, 3)
    curve_pts = curve.points
    num_samples = curve_pts.shape[0]
    
    dist_matrix = cdist(self.coors, curve_pts)
    collision_mask = np.any(dist_matrix < vdw_radius, axis=0)
    
    #actual_spacing = curve.length / (num_samples - 1)
    diffs = np.diff(curve_pts, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_len = np.sum(segment_lengths)
    actual_spacing = total_len / (num_samples - 1)
    
    collision_length = np.sum(collision_mask) * actual_spacing
    
    hit_atom_mask = np.any(dist_matrix < vdw_radius, axis=1)
    hit_atom_indices = np.where(hit_atom_mask)[0]
    
    return collision_length, hit_atom_indices
    
  def interaction_energy_with_curve(self, curve, energy_func=None, cutoff=10.0):
    """
    Calculates an interaction energy score for the curve.
    """
    curve_pts = curve.points
    dist_matrix = cdist(self.coors, curve_pts)
    
    mask = (dist_matrix < cutoff) & (dist_matrix > 0.1)
    r    = dist_matrix[mask]
    
    if len(r) == 0:
      return 0.0
    
    if energy_func is None:
      # Avoid division by zero with a small epsilon
      energy_func = lambda r: 1.0 / (r**2 + 0.1)
    
    energies = energy_func(r)
    total_energy = np.sum(energies)
    return total_energy / curve_pts.shape[0]    

  def parse_pdb_standard(self, lines, with_hetero=True, with_bb=True):
    atom_rows = []
    coors = []
    model_index = 0
    
    for line in lines:
      # Could parse header lines (TITLE, EXPDTA, etc.) if needed
      if line.startswith("TITLE"):
        self.title = line[10:].strip()
      elif line.startswith("EXPDTA"):
        self.method = line[10:].strip()
      elif "RESOLUTION" in line:
        parts = line.strip().split()
        for p in parts:
          try:
            self.resolution = float(p)
            break
          except Exception:
            pass            
            
      if line.startswith("MODEL"):
        model_index += 1
        coors = []
        continue

      if line.startswith("ENDMDL"):
        self.models[model_index] = np.array(coors, dtype=float)
        continue        
        
      if line.startswith("ATOM") or (with_hetero and line.startswith("HETATM")):
        try:
          # parse fields by PDB fixed columns
          atom_id   = int(line[6:11])
          atom_name = line[12:16].strip()
          res_name  = line[17:20].strip()
          chain_id  = line[21:22].strip()
          if chain_id == '': chain_id = 'A'
          res_num    = int(line[22:26])
          
          x = float(line[30:38])
          y = float(line[38:46])
          z = float(line[46:54])
          
          occupancy = float(line[54:60]) if line[54:60].strip() else 0.0
          b_factor = float(line[60:66]) if line[60:66].strip() else 0.0
          element = line[76:78].strip()

          atom_row = {
              'record': line[0:6].strip(),
              'atom_id': atom_id,
              'atom_name': atom_name,
              'res_name': res_name,
              'chain_id': chain_id,
              'res_num': res_num,
              'element': element,
              'occupancy': occupancy,
              'b_factor': b_factor
          }
          coors.append((x, y, z))
          if model_index < 2:   # There is NO MODEL (model_index==0) or we are in the FIRST MODEL (model_index==1)
            atom_rows.append(atom_row)

        except ValueError as e:
          print(f"Skipping malformed ATOM/HETATM line: {line.strip()} ({e})")
          continue

    self.atoms  = pd.DataFrame(atom_rows)
    self.coors  = np.array(coors, dtype=float)
    
    rows_for_df = {}  # Dictionary of rows to build a pd.DataFrame at the end
    self.idxs   = {}
    
    bb_atoms = ['N', 'CA', 'C', 'O']
    if with_bb:
      chain_ids = pd.unique(self.atoms['chain_id'])
      for (chain_id, res_name, res_num), res_df in self.atoms.groupby(['chain_id','res_name','res_num'], sort=False):
        if chain_id not in rows_for_df:
          rows_for_df[chain_id] = []
          
        res_bb = {a: None for a in bb_atoms}
        res_sc = []
        for i, row in res_df.iterrows():
          an = row['atom_name']
          if an in bb_atoms:
            res_bb[an] = i
          else:
            res_sc.append(i)  
            
        insert_residue = True
        '''
        insert_residue = False
        if all(v is not None for v in res_bb.values()):
          insert_residue = True
        else:
          if res_bb['O'] is None and all(v is not None for bba, v in res_bb.items() if bba != 'O'):
            res_bb['O'] = -1
            insert_residue = True
        '''
        if insert_residue:  
          bb_inds  = [res_bb[a] for a in bb_atoms]
          rows_for_df[chain_id].append({
            'chain_id': chain_id,
            'res_name': res_name,
            'res_num':  int(res_num),
            'bb_idxs': tuple(bb_inds),
            'sc_idxs': tuple(res_sc),  # snapshot for readability
          })

    for chain in rows_for_df:
      self.idxs[chain] = pd.DataFrame(rows_for_df[chain])

  def Read_pdb(self, pdb_file_path, chain_to_add_patches: bool | str = False, with_hetero = False, verbose = False):
    if not os.path.exists(pdb_file_path):
      raise FileNotFoundError(f"PDB file {pdb_file_path} not found")

    if verbose:
      print("Reading PDB file '{}'".format(pdb_file_path))
  
    try:
      with open(pdb_file_path, 'r') as f:
        lines = f.readlines()
        
      self.parse_pdb_standard(lines, with_hetero = with_hetero)
      if chain_to_add_patches:
        if isinstance(chain_to_add_patches, bool):   
          chain_to_set = next(iter(self.idxs))
        elif isinstance(chain_to_add_patches, str):
          chain_to_set = chain_to_add_patches 
        else:
          chain_to_set = None
          print("Warning: No chains found in PDB to add patches to.")          
          
        self.Set_active_chain(chain_to_set)
      self.calculate_geometrical_properties()       
       
    except Exception as e:
      #print("Error in reading PDB file '{}'".format(pdb_file_path))
      print(f"Error in reading PDB file '{pdb_file_path}': {e}")
      self.atoms = pd.DataFrame()
      self.coors = None
  
  def Write_pdb(self, filename, coors_to_write=None, atom_indices=None):
    if coors_to_write is None:
      coors_to_write = self.coors

    if atom_indices is not None:
        # If filtering by atoms
        atom_indices = np.asarray(atom_indices)
        # Slice coordinates if still full
        if coors_to_write is self.coors:
          coors_to_write = coors_to_write[atom_indices]
    else:
        #atom_indices = self.atoms.index  # write all
        # No atom_indices → use all
        atom_indices = self.atoms.index.to_numpy()
        
    if len(atom_indices) != len(coors_to_write):
      raise ValueError("Number of coordinates doesn't match number of selected atoms")

    with open(filename, 'w') as f:
      #f.write(f"HEADER    {self.title}\n")
      # Atom lines
      for idx, i in enumerate(atom_indices):
        atom = self.atoms.loc[i]
        x, y, z = coors_to_write[idx]
        # Format line in PDB fixed‑column format
        line = (
          f"{atom['record']:<6s}"
          f"{atom['atom_id']:5d}  "
          f"{atom['atom_name']:<4s}"
          #f"{atom['atom_name']:<4s} "
          #f"{atom['atom_name']:>4s} "
          f"{atom['res_name']:>3s} "
          f"{atom['chain_id']:1s}"
          f"{atom['res_num']:4d}    "
          f"{x:8.3f}{y:8.3f}{z:8.3f}"
          f"{atom['occupancy']:6.2f}{atom['b_factor']:6.2f}          "
          f"{atom['element']:>2s}"
        )
        f.write(line + "\n")
      f.write("END\n")
    print(f"Wrote the File '{filename}'.")

  def Write_termini_as_pdb(self, filename):
    termini_idxs = np.concatenate([self.Nt, self.Ct])
    self.Write_pdb(filename, atom_indices = termini_idxs)

  def Start_pdb_model_writer(self, filename, atom_indices=None):
    self._model_pdb_file = open(filename, 'w')
    self._model_pdb_filename = filename
    self._model_atom_indices = np.asarray(atom_indices) if atom_indices is not None else self.atoms.index.to_numpy()
    self._model_number = 1

    self._model_pdb_file.write(f"HEADER    {self.title}\n")
    print(f"Initialized model writing to '{filename}'")

  def Append_model(self, coors, model_number=None):
    if not hasattr(self, '_model_pdb_file'):
        raise RuntimeError("Call start_pdb_model_writer() first.")

    if model_number is None:
        model_number = self._model_number

    atom_indices = self._model_atom_indices

    if coors.shape != (len(atom_indices), 3):
        raise ValueError("Coordinate shape does not match selected atom indices")

    f = self._model_pdb_file
    f.write(f"MODEL     {model_number}\n")

    for i, atom_idx in enumerate(atom_indices):
        atom = self.atoms.loc[atom_idx]
        x, y, z = coors[i]

        line = (
            f"{atom['record']:<6s}"
            f"{atom['atom_id']:5d} "
            f"{atom['atom_name']:>4s} "
            f"{atom['res_name']:>3s} "
            f"{atom['chain_id']:1s}"
            f"{atom['res_num']:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{atom['occupancy']:6.2f}{atom['b_factor']:6.2f}          "
            f"{atom['element']:>2s}"
        )
        f.write(line + "\n")

    f.write("ENDMDL\n")
    print(f"Appended the Model [{self._model_number}] to the File '{self._model_pdb_filename}'.")
    self._model_number += 1  # increment model counter

  def Finish_pdb_model_writer(self):
    if hasattr(self, '_model_pdb_file'):
        self._model_pdb_file.write("END\n")
        self._model_pdb_file.close()
        print(f"Finished writing models to '{self._model_pdb_filename}'")
        del self._model_pdb_file

  #@staticmethod
  def create_from_dm(self, dm, title="Reconstructed from Distance Matrix"):
    """
    Creates a new Ensamble instance with coordinates reconstructed from 
    the input distance matrix (dm) using Metric Multidimensional Scaling (MDS).
    
    Default 'C' atoms are assigned to the new structure.

    :param dm: The NxN Euclidean distance matrix (NumPy array).
    :param title: The title to assign to the new Ensamble.
    :return: A new Ensamble instance with reconstructed coordinates, or None on failure.
    """
    N_atoms = dm.shape[0]
    try:
        # 1. Initialize and run the MDS solver
        mds = MDS(
            n_components=3, 
            dissimilarity='precomputed', 
            random_state=0, 
            n_init=1
        )
        reconstructed_coords = mds.fit_transform(dm)
        # 3. Create default atom descriptors (using 'C' as the default atom)
        default_atoms = {
            'record': ['ATOM'] * N_atoms,
            'atom_id': np.arange(1, N_atoms + 1),
            'atom_name': ['C'] * N_atoms,
            'res_name': ['RES'] * N_atoms,
            'chain_id': ['A'] * N_atoms,
            'res_num': [1] * N_atoms,
            'element': ['C'] * N_atoms,
            'occupancy': [1.0] * N_atoms,
            'b_factor': [0.0] * N_atoms,
        }
        self.atoms = pd.DataFrame(default_atoms)
        
        # 4. Assign the reconstructed coordinates
        self.coors = reconstructed_coords
        self.title = title
        
        print(f"New Ensamble created with {N_atoms} 'C' atoms from distance matrix.")
        
    except Exception as e:
        print(f"An unexpected error occurred during MDS reconstruction: {e}")
        return None
  
  def extract_sparse_substructure(self, cutoff_distance):
    """
    Identifies a subset of atoms that are all mutually separated by a 
    distance greater than the cutoff, and returns a new distance matrix 
    and atom indices for this subset.
    
    Uses a greedy heuristic to select the subset.

    :param cutoff_distance: The minimum required distance between any two selected atoms.
    :return: Tuple (filtered_dm, selected_indices) or (None, None) on failure.
    """
    if self.dm is None:
        print("Error: Distance matrix (self.dm) has not been calculated.")
        return None, None
        
    N_atoms = self.dm.shape[0]
    if N_atoms == 0:
        return None, None

    # 1. Initialization
    all_indices = np.arange(N_atoms)
    selected_indices = []
    
    # We start with the first atom and greedily add any atom that meets the distance criteria

    # 2. Greedy Selection Process
    for current_idx in all_indices:
        # Check if this atom is already selected
        if current_idx in selected_indices:
            continue

        # Check distance against all *already selected* atoms
        is_valid = True
        for selected_idx in selected_indices:
            # The distance matrix is symmetric; check distance once
            distance = self.dm[current_idx, selected_idx]
            
            if distance <= cutoff_distance:
                is_valid = False
                break  # This atom is too close to an already selected one
        
        # If valid (distant enough from all current selections), add it to the set
        if is_valid:
            selected_indices.append(current_idx)

    selected_indices = np.array(selected_indices, dtype=int)
    
    # 3. Extract the new distance matrix (DM)
    # Use NumPy indexing to slice both rows and columns simultaneously
    filtered_dm = self.dm[selected_indices, :][:, selected_indices]
    
    print(f"Extracted subset: Original atoms: {N_atoms}, Selected atoms: {len(selected_indices)}")
    print(f"All selected atoms are mutually separated by > {cutoff_distance:.2f} Å.")
    
    return filtered_dm, selected_indices

  # --- New Ensamble Constructor using the above method ---
  def create_sparse_from_dm(self, cutoff_distance):
    """
    Creates a new Ensamble instance containing only the sparse subset of atoms
    whose mutual distances are greater than cutoff_distance.
    """
    filtered_dm, selected_indices = self.extract_sparse_substructure(cutoff_distance)

    if filtered_dm is None:
        return None

    # 1. Use the static method for reconstruction
    # We need to create a new Ensamble instance using the static method
    #from . import Ensamble # Assuming Ensamble is imported or available via context
    new_ensemble = Ensamble()
    new_ensemble.create_from_dm(filtered_dm, title=f"Sparse Structure from {self.title} (Cutoff > {cutoff_distance:.2f} A)")
    
    # 2. Update atom descriptors for the new Ensamble
    if new_ensemble:
        # Copy atom data only for the selected indices
        new_ensemble.atoms = self.atoms.iloc[selected_indices].reset_index(drop=True)
        # Re-index the atom IDs if needed (for PDB writing)
        new_ensemble.atoms['atom_id'] = np.arange(1, len(selected_indices) + 1)
        
    return new_ensemble
    
  def Spherize(self, target_radius=None):
    self.calculate_geometrical_properties()
    self.calculate_physical_properties()
    
    centered_coords = self.coors - self.center
    projected_coords = np.dot(centered_coords, self.evectors)

    # Calculate scaling factors
    scales = 1.0 / (np.sqrt(self.evalues) + 1e-8)
    scales = scales / np.mean(scales)
    # Apply the scaling in the Principal Axis space
    spherized_projected = projected_coords * scales

    # Back Rotation
    self.coors = np.dot(spherized_projected, self.evectors.T) + self.center
    
  def Spherify(self, type = 'extend'):
    self.Orient()
    if   type == 'eigen':
      self.calculate_physical_properties(True)
      values = np.sqrt(self.evalues)
    elif type == 'dim':    
      self.calculate_geometrical_properties()
      values = self.dim
    elif type == 'extend':    
      values = np.std(self.coors, axis=0)
      
    scales = 1.0/ values    
    self.scales = scales / np.mean(scales)
    print("SCALES", self.scales)
    self.coors = self.coors * self.scales

  def Expand(self, original_center):
    """
    Reverses the Spherify process to restore the original rod-like shape.
    Assumes self.scales, self.U, and self.R_p were saved during Spherify/Orient.
    """
    self.coors = self.coors - original_center
    
    # 1. Reverse Scaling
    # We divide by the scales used in Spherify
    self.coors = self.coors / self.scales

    # 2. Reverse 'In-frame alignment' (U)
    # Since U is a rotation matrix, its inverse is its transpose
    if hasattr(self, 'U'):
        self.coors = self.coors @ self.U.T

    # 3. Reverse 'Principal frame alignment' (R_p)
    # We rotate back from the principal frame to the world frame
    if hasattr(self, 'R_p'):
        self.coors = self.coors @ self.R_p.T

    # 4. Reverse Centering
    # Move the restored rod back to its original position in space
    self.coors = self.coors + original_center
    
    # 5. Refresh properties
    self.calculate_geometrical_properties()
    print("Structure expanded back to original dimensions.")   
      
def Create_protein(seq, direction = "NC"):
  print("Creating Protein with the sequence:")
  print(seq)
  pr = None
  
  growing_seq_display = ""
  print("Building sequence:")
  for aan in seq:
    if aan in s2t:
      #print(f"Adding {aan}")
      add_res = s2t[aan]
      if pr:
        pr.Add_residue(add_res, direction)
      else:
        res_file_path = os.path.join(base_dir, res_dir, f"{add_res}.pdb")
        pr = Ensamble(res_file_path, add_patches=True)
        
      growing_seq_display += aan
      print(f"\r{growing_seq_display}", end="")
      sys.stdout.flush()
    else:
      print(f"Found non-standard Amino Acid '{aan}'")
      return None
  print()
  return pr
  
patch_filename = "for_patches.pdb"
patch_path = os.path.join(base_dir, lib_dir, patch_filename)
patchMol = Ensamble(patch_path, chain_to_add_patches = True)