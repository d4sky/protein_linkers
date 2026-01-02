import numpy as np

from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
#from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Curve():
  def __init__(self, N = 100):
    self.N_vertices  = N
    self.t           = []   # List of t parameters (? 0 to 1)
    self.points      = []   # Points on the CURVE = smoothed LINE from above line, on which the Objects/Molecules will be projected, i.e. rotated and translated
    self.tangents    = []   
    self.normals     = []   
    self.lengths     = []   # Cumulative length at each point
    self.edges       = []   # Indices or vectors connecting point
    
    self.length  = 0.0     # Total length
    self.delta   = 0.0     # Distance between vertices
    
    
    self.length  = 0.0
    self.delta   = 0.0
    
    self._s_to_t = None
    self.spline_params = None

  def create_from_line(self, P1, P2):
    R1 = np.array(P1)
    R2 = np.array(P2)
    
    # 1. Basic Geometry
    R12 = R2 - R1
    self.length = np.linalg.norm(R12)
    self.delta = self.length / (self.N_vertices - 1)
    self.t = np.linspace(0, 1, self.N_vertices).tolist()
    
    Ru = R12 / self.length
    point = [R1 + i*Ru*self.delta for i, t_val in enumerate(self.t)]
    self.points = np.array(point)

  def create_from_points(self, points):
    """
    Initializes the Curve using a sequence of points (e.g., a Meridian).
    Calculates tangents, normals, and cumulative arc length.
    """
    self.points = np.array(points)
    self.N_vertices = len(self.points)
    
    # 1. Calculate cumulative lengths and total length
    diffs = np.diff(self.points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    self.lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    self.length = self.lengths[-1]
    
    # 2. Parameters (normalized 0 to 1 based on arc length)
    if self.length > 0:
        self.t = self.lengths / self.length
        self.delta = self.length / (self.N_vertices - 1)
    else:
        self.t = np.zeros(self.N_vertices)
        
    # 3. Calculate Tangents (T)
    # Using central differences for smoothness
    tangents = np.zeros_like(self.points)
    tangents[1:-1] = (self.points[2:] - self.points[:-2])
    # Handle endpoints
    tangents[0]  = self.points[1] - self.points[0]
    tangents[-1] = self.points[-1] - self.points[-2]
    
    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    self.tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)

    # 4. Calculate Normals (N) and Edges
    self.normals = []
    self.edges = []
    for i in range(self.N_vertices):
        # Edge index pairs
        if i < self.N_vertices - 1:
            self.edges.append([i, i + 1])
            
        # Normal vector (principal curvature direction)
        # For a simple implementation: cross tangent with a fixed reference
        # or use the derivative of the tangent
        if i < self.N_vertices - 1:
            n_vec = self.tangents[i+1] - self.tangents[i]
        else:
            n_vec = self.tangents[i] - self.tangents[i-1]
            
        n_mag = np.linalg.norm(n_vec)
        if n_mag > 1e-6:
            self.normals.append(n_vec / n_mag)
        else:
            # Fallback if curve is straight: pick any vector perpendicular to tangent
            ref = np.array([1, 0, 0]) if abs(self.tangents[i][0]) < 0.9 else np.array([0, 1, 0])
            n_perp = np.cross(self.tangents[i], ref)
            self.normals.append(n_perp / np.linalg.norm(n_perp))

    self.normals = np.array(self.normals)  
    self.fit_spline()

  def Write_pdb(self, filename):
    with open(filename, 'w') as f:
      f.write(f'REMARK Created by Curve class\n')
        
      # Write ATOM records for points
      for i, vertex in enumerate(self.points):
        #f.write(f'ATOM  {i+1:4}  C   UNK A   1     {vertex[0]:7.3f}{vertex[1]:7.3f}{vertex[2]:7.3f}  1.00  0.00           C\n')
        #out_line = PDB_line('C', "NOT NEEDE", i+1, "UNK", 'A', 1, ' ', vertex)
        out_line = "ATOM  {:>5d} {:<4s} {:>3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(i+1, 'C', 'UNK', 'A', 1, ' ', vertex[0], vertex[1], vertex[2], 1.00, 0.00)
        f.write(out_line)
      
      if not self.edges and len(self.points) >= 2:
        self.edges = [(i, i+1) for i in range(len(self.points) - 1)]

      # Write CONECT records
      for edge in self.edges:
        f.write(f'CONECT {edge[0]+1:4}{edge[1]+1:4}\n')    
          
      f.write('END\n')

  def Write_tangents(self, filename, r=1.0):
    """
    Writes a PDB file showing tangent vectors as lines (bonds) starting from each point,
    using CONECT records. Each vector is scaled by length r.
    """
    with open(filename, 'w') as f:
      f.write('REMARK Tangent vectors visualized as bonds\n')

      atom_index = 1
      for i, (P, T) in enumerate(zip(self.points, self.tangents)):
        EP = P + r*T
        # Write the two atoms
        f.write("ATOM  {:>5d} {:<4s} {:>3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
            atom_index, 'C', 'TAN', 'A', 1, ' ', P[0], P[1], P[2], 1.00, 0.00))
        atom_index += 1

        f.write("ATOM  {:>5d} {:<4s} {:>3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
            atom_index, 'O', 'TAN', 'A', 1, ' ', EP[0], EP[1], EP[2], 1.00, 0.00))
        
        # Write CONECT between the two
        f.write(f"CONECT {atom_index-1:4}{atom_index:4}\n")
        atom_index += 1

      f.write('END\n')

  def Write_normals(self, filename, r=1.0):
    """
    Writes a PDB file showing tangent vectors as lines (bonds) starting from each point,
    using CONECT records. Each vector is scaled by length r.
    """
    with open(filename, 'w') as f:
      f.write('REMARK Tangent vectors visualized as bonds\n')

      atom_index = 1
      for i, (P, T) in enumerate(zip(self.points, self.normals)):
        EP = P + r*T
        # Write the two atoms
        f.write("ATOM  {:>5d} {:<4s} {:>3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
            atom_index, 'C', 'NRM', 'A', 1, ' ', P[0], P[1], P[2], 1.00, 0.00))
        atom_index += 1

        f.write("ATOM  {:>5d} {:<4s} {:>3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
            atom_index, 'N', 'NRM', 'A', 1, ' ', EP[0], EP[1], EP[2], 1.00, 0.00))
        
        # Write CONECT between the two
        f.write(f"CONECT {atom_index-1:4}{atom_index:4}\n")
        atom_index += 1

      f.write('END\n')

  def value(self, t):
    """
    Return a point on the curve at parameter t.
    Parameters: - t: Parameter value in the range [0, 1].
    Returns:    - point: Point on the curve at parameter t.
    """
    if self.spline_params is None:
      raise ValueError("Spline parameters have not been computed.")

    # Evaluate the spline at the parameter t
    point = splev(t, self.spline_params)
    return np.array(point)
    
  def get_t_param(self, force_rebuild: bool = False):
    """
    Returns a function that maps arc length s → parameter t
    using self.t and self.points.
    Assumes:
        - self.t is a 1D array of length N
        - self.points is a (N, 3) NumPy array
    """
    #if force_rebuild or not hasattr(self, "_s_to_t"):
    if force_rebuild or self._s_to_t is None:
      # Create interpolation function: arc_length → t
      self._s_to_t = interp1d(self.lengths, self.t, kind='linear', fill_value='extrapolate')
    return self._s_to_t

  def order_points(self, eTol = 0.1):
    """
    Sort points to minimize the total distance traveled along the curve.
    """
    points = np.array(self.points)
    N = len(points)

    if N <= 1:
        return

    sorted_points = [points[0]]
    remaining_points = list(points[1:])
    
    current_point = points[0]
    
    while remaining_points:
        # Find the closest point to the current point
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        closest_index = np.argmin(distances)
        closest_point = remaining_points.pop(closest_index)

        # Check if the distance to the closest point is greater than the threshold
        if distances[closest_index] > eTol:
          sorted_points.append(closest_point)
          current_point = closest_point

    self.points = np.array(sorted_points)
    
  def sort_points__(self, ref_point, eTol = 0.1):
    print("Original points:", self.points)
    
    points = np.array(self.points)
    N = len(points)

    if N <= 1:
        return
    
    # Calculate distances from ref_point to each point in self.points
    distances = np.linalg.norm(points - ref_point, axis=1)

    # Sort indices of points based on distances
    sorted_indices = np.argsort(distances)

    # Initialize variables
    sorted_points = []

    # Find the index of the point closest to ref_point
    closest_index = sorted_indices[0]

    # Add points before the closest point
    for idx in range(closest_index):
        if distances[idx] <= eTol:
            sorted_points.append(points[idx])

    # Add the closest point
    sorted_points.append(points[closest_index])

    # Add points after the closest point
    for idx in range(closest_index + 1, N):
        if distances[idx] <= eTol:
            sorted_points.append(points[idx])

    self.points = sorted_points
    print("Sorted points:", self.points)
    
  def sort_points(self, inp_point, dTol=0.1, hTol = float("inf")):
    ref_point = np.array(inp_point)  
    points = np.array(self.points)
    if len(points) <= 1: return
    
    distances = np.linalg.norm(points - ref_point, axis=1)
    closest_index = np.argmin(distances)
    remaining_points = list(points)
    current_point    = remaining_points.pop(closest_index)
    
    lpoints, lvec = [], np.zeros(3)
    rpoints, rvec = [], np.zeros(3)

    while remaining_points:
      if lpoints:
        ldistances = np.linalg.norm(remaining_points - lpoints[-1], axis=1)
        lclosest_dist  = min(ldistances)
        lclosest_index = np.argmin(ldistances)

        if rpoints:
          rdistances = np.linalg.norm(remaining_points - rpoints[-1], axis=1)
        else:
          rdistances = np.linalg.norm(remaining_points - ref_point, axis=1)
        rclosest_dist  = min(rdistances)
        rclosest_index = np.argmin(rdistances)
          
        if lclosest_dist < rclosest_dist:
          closest_point = remaining_points.pop(lclosest_index)
          if lclosest_dist > dTol and lclosest_dist < hTol:
            lpoints.append(np.array(closest_point))
        else:
          closest_point = remaining_points.pop(rclosest_index)
          if rclosest_dist > dTol and rclosest_dist < hTol:
            rpoints.append(np.array(closest_point))
      else:
        ldistances = np.linalg.norm(remaining_points - ref_point, axis=1)
        closest_dist  = min(ldistances)
        lclosest_index = np.argmin(ldistances)
        closest_point = remaining_points.pop(lclosest_index)
        if closest_dist > dTol and closest_dist < hTol:
          lpoints.append(np.array(closest_point))
        
    while abs(len(lpoints) - len(rpoints)) > 1:
      if len(lpoints) > len(rpoints):
        rpoints.append(lpoints.pop())
      else:
        lpoints.append(rpoints.pop())
    #self.points = np.array(lpoints + [ref_point] + rpoints)
    self.points = np.array(lpoints[::-1] + [ref_point] + rpoints)

  def calculate_tangents(self):
    """
    Calculates normalized tangents (T) for the 3D polyline stored in self.points 
    using central finite difference approximation.
    """
    if len(self.points) < 3:
        # Need at least 3 points for central difference approximation
        print("Warning: Cannot calculate Frenet frame with less than 3 points.")
        self.tangents = np.zeros_like(self.points)
        self.normals = np.zeros_like(self.points)
        return

    points = self.points
    N_pts = points.shape[0]
    
    # Calculate Tangents (First Derivative approx: T ~ P_i+1 - P_i-1)
    # Central difference for interior points
    tangents = points[2:] - points[:-2]
    
    # Handle endpoints with forward/backward difference
    tangent_start = points[1] - points[0]
    tangent_end = points[N_pts-1] - points[N_pts-2]
    
    full_tangents = np.vstack([tangent_start, tangents, tangent_end])
    
    # Normalize Tangents
    tangent_norms = np.linalg.norm(full_tangents, axis=1, keepdims=True)
    # Avoid division by zero
    tangent_norms[tangent_norms == 0] = 1.0
    self.tangents = full_tangents / tangent_norms

  def calculate_frenet_normals_from_points(self):
    """
    Calculates normalized normals (N, principal normal) 
    """
    if len(self.points) < 3:
        # Need at least 3 points for central difference approximation
        print("Warning: Cannot calculate Frenet frame with less than 3 points.")
        self.tangents = np.zeros_like(self.points)
        self.normals = np.zeros_like(self.points)
        return

    points = self.points
    # Calculate Curvature Vector K (Second Derivative approx: K ~ P_i+1 - 2*P_i + P_i-1)
    # K is the unnormalized Principal Normal vector
    curvature_vectors = points[2:] - 2 * points[1:-1] + points[:-2]
    
    # Handle endpoints for K (using neighbor's value for stability)
    K_start = curvature_vectors[0]
    K_end = curvature_vectors[-1]
    
    full_K = np.vstack([K_start, curvature_vectors, K_end])
    
    # Normalize to get the Principal Normal N
    K_norms = np.linalg.norm(full_K, axis=1, keepdims=True)
    N = np.zeros_like(full_K)
    
    # Only normalize where the curvature is non-zero
    non_zero_indices = K_norms.flatten() > 1e-9 
    N[non_zero_indices] = full_K[non_zero_indices] / K_norms[non_zero_indices]
    self.normals = N

  def calculate_Frenet_normals_from_tangents(self):
    dT = np.zeros_like(self.tangents)
    dT[1:-1] = (self.tangents[2:] - self.tangents[:-2]) / (2 * (self.t[1] - self.t[0]))

    # For endpoints (forward/backward difference)
    dT[0] = (self.tangents[1] - self.tangents[0]) / (self.t[1] - self.t[0])
    dT[-1] = (self.tangents[-1] - self.tangents[-2]) / (self.t[1] - self.t[0])

    # Normalize dT to get unit normals
    norms = np.linalg.norm(dT, axis=1, keepdims=True)
    self.normals = dT / norms

  def calculate_plane_normals(self, a, b, c, d, dTol=1e-6):
    plane_normal = np.array([a, b, c])
    # Calculate normals for the intersection points
    for i in range(len(self.points)):
      if i == 0:  # First point
        tangent = np.array(self.points[1]) - np.array(self.points[0])
      elif i == len(self.points) - 1:  # Last point
        tangent = np.array(self.points[-1]) - np.array(self.points[-2])
      else:  # Middle points
        tangent = np.array(self.points[i + 1]) - np.array(self.points[i - 1])

      tangent_norm = np.linalg.norm(tangent)
      tangent = tangent / tangent_norm  # Normalize the tangent
      self.tangents.append(tangent.tolist())

      normal = np.cross(plane_normal, tangent)
      normal_norm = np.linalg.norm(normal)
      if normal_norm != 0:
        normal = normal / normal_norm  # Normalize the normal
      else:
        normal = np.array([0.0, 0.0, 0.0])
      self.normals.append(normal.tolist())      

  def fit_spline(self):
    points = self.points.T

    # Debugging output to verify the shape
    #print(f"Points shape after transpose: {points.shape}")

    if points.shape[0] != 3:
        raise ValueError("Points array should have 3 dimensions (x, y, z).")
    try:
        tck, u = splprep(points, s=0)
        self.spline_params = tck
    except Exception as e:
        print(f"Error in splprep: {e}")
        raise
        
  def calculate_tangent_normal_from_spline(self, t):
    if t < 0 or t > 1:
        #raise ValueError("Parameter t must be in the range [0, 1].")
        pass

    if self.spline_params is None:
        raise ValueError("Spline parameters have not been computed.")

    # Evaluate the first derivative of the spline at the parameter t to get the tangent
    tangent = splev(t, self.spline_params, der=1)
    tangent = np.array(tangent)
    tangent /= np.linalg.norm(tangent)  # Normalize the tangent

    # Evaluate the second derivative of the spline at the parameter t to get the normal
    # The normal can be calculated using the derivative of the tangent vector.
    normal = splev(t, self.spline_params, der=2)
    normal = np.array(normal)
    normal -= normal.dot(tangent) * tangent  # Ensure normal is perpendicular to tangent
    normal /= np.linalg.norm(normal)  # Normalize the normal

    return tangent, normal

  def distances_to_other_(self, other_curve, resolution=500, dTol=0.1):
    """
        Find local minima in distances between two curves.

        Parameters:
        - other_curve: Another Curve instance to compare with.
        - resolution: Number of points to sample along each curve.
        - dTol: Distance tolerance to identify local minima.

        Returns:
        - local_minima: List of tuples (t1, t2, distance) where t1 is a parameter for this curve and t2 is for the other curve.
    """
    # Generate parameter values
    t_values = np.linspace(0, 1, resolution)

    initial_minima = []
    for i in range(1, resolution - 1):
        t1 = t_values[i]
        point1 = self.value(t1)
        
        # Calculate distances from point1 to all points on the other curve
        distances = np.linalg.norm(point1 - np.array([other_curve.value(t_val) for t_val in t_values]), axis=1)
        best_dist = np.min(distances)
        best_t    = t_values[np.argmin(distances)]
        initial_minima.append((t1, best_t, best_dist))
        
    #return np.array(initial_minima)
    #for fuck in (initial_minima):
    #  print(fuck)
    
    # Find local minima in the initial_minima list
    local_minima = []
    for i in range(1, len(initial_minima) - 1):
        prev_dist = initial_minima[i - 1][2]
        curr_dist = initial_minima[i][2]
        next_dist = initial_minima[i + 1][2]
        
        if curr_dist < prev_dist and curr_dist < next_dist:
            local_minima.append(initial_minima[i])
    
    return local_minima

  def distances_to_other(self, other_curve, resolution=20, dTol=1e-3):
    """
    Find local minima in distances between two curves.

    Parameters:
    - other_curve: Another Curve instance to compare with.
    - resolution: Number of points to sample along each curve.
    - dTol: Distance tolerance to identify local minima.

    Returns:
    - local_minima: List of tuples (t1, t2, distance) where t1 is a parameter for this curve and t2 is for the other curve.
    """
    # Generate parameter values
    t_values = np.linspace(0, 1, resolution)

    # Sample points on both curves
    self_points = np.array([self.value(t) for t in t_values])
    other_points = np.array([other_curve.value(t) for t in t_values])

    # Compute pairwise distances between sampled points
    distances = cdist(self_points, other_points)

    # Find initial local minima in the distance matrix
    initial_minima = []
    for i in range(1, resolution - 1):
        for j in range(1, resolution - 1):
            if distances[i, j] < distances[i - 1, j] and distances[i, j] < distances[i + 1, j] and \
               distances[i, j] < distances[i, j - 1] and distances[i, j] < distances[i, j + 1]:
                initial_minima.append((t_values[i], t_values[j], distances[i, j]))

    # Use local minimization around the initial minima
    def distance_func(t):
        point1 = self.value(t[0])
        point2 = other_curve.value(t[1])
        return np.linalg.norm(point1 - point2)

    refined_minima = []
    for t1, t2, _ in initial_minima:
        result = minimize(distance_func, [t1, t2], bounds=[(0, 1), (0, 1)], tol=dTol)
        if result.success:
            refined_t1, refined_t2 = result.x
            refined_distance = distance_func(result.x)
            if refined_distance < dTol:
              refined_minima.append((refined_t1, refined_t2, refined_distance))

    #return refined_minima
    
    final_minima = []
    for i, (t1, t2, dist) in enumerate(refined_minima):
      is_close = False
      for j, (t1_, t2_, dist_) in enumerate(final_minima):
        if np.linalg.norm([t1 - t1_, t2 - t2_]) < dTol:
          is_close = True
          # Further refine the minima if they are too close
          mid_t1 = (t1 + t1_) / 2
          mid_t2 = (t2 + t2_) / 2
          result = minimize(distance_func, [mid_t1, mid_t2], bounds=[(0, 1), (0, 1)], tol=dTol)
          if result.success:
            refined_t1, refined_t2 = result.x
            refined_distance = distance_func(result.x)
            final_minima[j] = (refined_t1, refined_t2, refined_distance)
          break
      if not is_close:
        final_minima.append((t1, t2, dist))

    return final_minima

  def distance_to_point(self, point, t):
    curve_point = np.array(self.value(t))
    target_point = np.array(point)
    distance = np.linalg.norm(curve_point - point)
    return distance

  def find_closest_t(self, point, verbose = True):
    def distance_to_point(t):
      curve_point = np.array(self.value(t[0]))
      return np.linalg.norm(curve_point - point)

    initial_guesses = [0.0, 0.25, 0.5, 0.75, 1.0]
    #initial_guesses = np.linspace(0.0, 1.0, 100)
    best_t = None
    min_distance = float('inf')
    for x0 in initial_guesses:
        result = minimize(distance_to_point, x0=x0, bounds=[(0, 1)], method='L-BFGS-B')
        #result = minimize(distance_to_point, x0=[x0], bounds=[(0, 1)], method='TNC')
        #result = minimize(distance_to_point, x0=x0, bounds=[(0, 1)], method='TNC')
        #result = minimize(distance_to_point, x0=[x0], bounds=[(0, 1)], method='SLSQP')
    
        #result = minimize(distance_to_point, x0=x0)
        if result.success:
            distance = distance_to_point([result.x[0]])
            #distance = distance_to_point(result.x[0])
            #print(distance, result.x[0])
            if distance < min_distance:
                min_distance = distance
                best_t = result.x[0]

    if best_t is None:
        raise ValueError("Optimization failed for all initial guesses.")

    '''
    new_best_t = None
    new_min_distance = float('inf')
    for t in np.linspace(0.0, 1.0, 1000):
      new_distance = np.linalg.norm(self.value(t) - point)
      if new_distance < new_min_distance:
        new_min_distance = new_distance    
        new_best_t = t
    '''
    
    t_values = np.linspace(0.0, 1.0, 1000)
    curve_points = np.array([self.value(t) for t in t_values])
    distances = np.linalg.norm(curve_points - point, axis=1)

    min_index = np.argmin(distances)
    new_best_t = t_values[min_index]
    new_min_distance = distances[min_index]

    if verbose:    
      print("MY BEST", new_best_t, new_min_distance, self.value(new_best_t), np.linalg.norm(self.value(new_best_t) - point))
      print("YOUR BEST", best_t, min_distance, self.value(best_t), np.linalg.norm(self.value(best_t) - point))
      
    if min_distance < new_min_distance:
      return best_t
    else:
      return new_best_t

  def find_closest_limited_t(self, point, t_limit=0.0, verbose=True):
    t_values = np.linspace(t_limit, 1.0, 100)
    curve_points = np.array([self.value(t) for t in t_values])
    distances = np.linalg.norm(curve_points - point, axis=1)

    min_idx  = np.argmin(distances)
    best_t   = t_values[min_idx]
    min_dist = distances[min_idx]      
      
    def distance_to_point(t):
      # t is passed as a list/array by minimize
      curve_point = np.array(self.value(t[0]))
      return np.linalg.norm(curve_point - point)

    results = minimize(
        distance_to_point, 
        x0=[best_t], 
        bounds=[(t_limit, 1.0)], 
        method='L-BFGS-B',
        tol=1e-4 # Protein atoms don't need 1e-9 precision
    )

    if results.success and results.fun < min_dist:
        final_t    = results.x[0]
        final_dist = results.fun
    else:
        final_t    = best_t
        final_dist = min_dist

    if verbose:    
        print(f"SEARCH LIMIT: t >= {t_limit}")
        print(f"Systematic search found t={best_t:.4f} (d={min_dist:.3f})")
        print(f"Final polished t={final_t:.4f} (d={final_dist:.3f})")
        
    return final_t

  def find_closest_limited_distance(self, point, t_param):
    t_min = self.find_closest_limited_t(point, t_param, False)
    d_min = self.distance_to_point(point, t_min)
    return d_min, t_min
    
  def find_closest_distance(self, point):
    t_min = self.find_closest_t(point, False)
    d_min = self.distance_to_point(point, t_min)
    return d_min, t_min
    
  def Add_edge(self, insP1, insP2):
    self.points.append(insP1)
    self.points.append(insP2)
    Nlast = len(self.points)
    self.edges.append((Nlast-2, Nlast-1))    

  def Angle_project_polar(self, t, ar, afi):
    """
    Projects a point at distance 'ar' along the normal vector,
    then rotates it by angle 'afi' (radians) around the tangent vector,
    starting from arc-length position 's' along the curve.
    
    Returns: np.array([x, y, z])
    """
    # Step 1: get t parameter from arc length
    #get_t = self.get_t_param()
    #t = float(get_t(s))  # ensure scalar

    # Step 2: interpolate r0, rr (tangent), rh (normal)
    r0 = np.array([np.interp(t, self.t, self.points[:, i]) for i in range(3)])
    rr = np.array([np.interp(t, self.t, self.tangents[:, i]) for i in range(3)])
    rh = np.array([np.interp(t, self.t, self.normals[:, i]) for i in range(3)])

    # Step 3: normalize tangent and normal
    rr = rr / np.linalg.norm(rr)
    rh = rh / np.linalg.norm(rh)

    # Step 4: project point along normal
    P = r0 + ar * rh

    # Step 5: rotate point P around r0 + rr using Rodrigues' formula

    # Vector from r0 to P
    v = P - r0

    # Axis (tangent) is rr
    k = rr  # already normalized

    # Rodrigues' rotation formula
    cos_a = np.cos(afi)
    sin_a = np.sin(afi)

    rotated = (v * cos_a +
      np.cross(k, v) * sin_a +
      k * np.dot(k, v) * (1 - cos_a)
    )

    final_point = r0 + rotated
    return final_point

  def Angle_project(self, t, P0, P1, axis = 'z', get_add_angle = None):
    """
    Projects a point at P1 relative to P0 on the curve at the positiion t
    Returns: np.array([x, y, z])
    """
    self.get_t = self.get_t_param() #!!! Is necessary to get the whole length of the curve  
    x, y, z = P1
    if   axis == 'x':
      diff = x - P0[0]
      t = self.get_t(diff)
      ar = np.hypot(y, z)
      afi = np.arctan2(z, y)
    elif axis == 'y':
      diff = y - P0[0]
      t = self.get_t(diff)
      ar = np.hypot(x, z)
      afi = np.arctan2(x, z)
    else: # Default 'z'
      diff = z - P0[0]
      t = self.get_t(diff)
      ar = np.hypot(x, y)
      afi = np.arctan2(y, x)
     
    if get_add_angle is not None:
      afi = afi + get_add_angle(diff)
      
    # Step 1: get t parameter from arc length
    #get_t = self.get_t_param()
    #t = float(get_t(s))  # ensure scalar

    # Step 2: interpolate r0, rr (tangent), rh (normal)
    r0 = np.array([np.interp(t, self.t, self.points[:, i]) for i in range(3)])
    rr = np.array([np.interp(t, self.t, self.tangents[:, i]) for i in range(3)])
    rh = np.array([np.interp(t, self.t, self.normals[:, i]) for i in range(3)])

    # Step 3: normalize tangent and normal
    rr = rr / np.linalg.norm(rr)
    rh = rh / np.linalg.norm(rh)

    # Step 4: project point along normal
    P = r0 + ar * rh

    # Step 5: rotate point P around r0 + rr using Rodrigues' formula

    # Vector from r0 to P
    v = P - r0

    # Axis (tangent) is rr
    k = rr  # already normalized

    # Rodrigues' rotation formula
    cos_a = np.cos(afi)
    sin_a = np.sin(afi)

    rotated = (v * cos_a +
               np.cross(k, v) * sin_a +
               k * np.dot(k, v) * (1 - cos_a))

    final_point = r0 + rotated
    return final_point

class Trefoil_knot(Curve):
  def __init__(self, radius=50.0, N=100, eps = 0.01):
    Curve.__init__(self, N)
    
    t = np.linspace(0.0, 2*PI*(1.0 + eps), N)
    self.t  = t 
        
    sin1 = np.sin(t)
    cos1 = np.cos(t)
    sin2 = np.sin(2.0*t)
    cos2 = np.cos(2.0*t)
    sin3 = np.sin(3.0*t)
    cos3 = np.cos(3.0*t)
  
    x =  radius*(2.0*sin2 + sin1)
    y =  radius*(cos1 - 2.0*cos2)
    z = -radius*(sin3)
    self.points   = np.column_stack((x, y, z)) 

    # Tangenty
    xd =  radius*(4.0*cos2 + cos1)
    yd =  radius*(4.0*sin2 - sin1)
    zd = -radius*(3*cos3)
    
    self.tangents = np.column_stack((xd, yd, zd))  
    # Normalize each tangent
    norms = np.linalg.norm(self.tangents, axis=1, keepdims=True)
    self.tangents = self.tangents / norms    

    # Pomocny vektor - Normals to the curve in a Frenet style
    xh = (18.0*np.sin(8.0*t) + 9.0*np.sin(7.0*t) + 2.0*np.sin(5.0*t) + 94.0*np.sin(4.0*t) - 324.0*sin2 + 69.0*sin1)
    yh =-(18.0*np.cos(8.0*t) - 9.0*np.cos(7.0*t) + 2.0*np.cos(5.0*t) - 94.0*np.cos(4.0*t) - 324.0*cos2 - 69.0*cos1)
    zh = (18.0*(2.0*np.sin(6.0*t) + 17.0*sin3))
    self.normals = np.column_stack((xh, yh, zh))  

    # Normalize each Helper Vector
    norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
    self.normals = self.normals / norms    
    
    if len(self.t) != len(self.points):
      raise ValueError("self.t and self.points must have the same length")

    # Compute distances between consecutive points
    diffs = np.diff(self.points, axis=0)          # shape (N-1, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1)   # shape (N-1,)

    # Cumulative arc length
    arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # shape (N,)

    self.lengths = arc_lengths
    # Save total length
    self.length = arc_lengths[-1]

def V3_mode_shape(s: np.ndarray, L: float = 1.0, normalize: bool = True) -> np.ndarray:
  """
    Calculates the spatial mode shape V3(s) for the first elastic bending mode
    the n=3 mode) of a uniform elastic rod with free-free boundary conditions.
    This version operates on NumPy arrays for vectorized calculation.
  """
  # k is the wave number (spatial frequency)
  k = BETA_3 / L

  # Calculate the boundary condition constant A (fixed for this mode)
  A_numerator = np.cosh(BETA_3) - np.cos(BETA_3)
  A_denominator = np.sinh(BETA_3) - np.sin(BETA_3)
  A = A_numerator / A_denominator
    
  # Calculate the unnormalized mode shape V(s) - vectorized operation
  V_unnormalized = (np.cosh(k * s) + np.cos(k * s)) - A * (np.sinh(k * s) + np.sin(k * s))
    
  if normalize:
    # Find the value at the end s=0 for normalization (safest method)
    # Note: We must call V3_mode_shape with a scalar 0.0 and normalize=False 
    # to get the magnitude of the end deflection.
    V_at_end = V3_mode_shape(np.array([0.0]), L, False)[0]
    
    # Normalize the array:
    V_normalized = V_unnormalized / abs(V_at_end)
    return V_normalized
  else:
    return V_unnormalized
        
class V3_mode_rod(Curve):
  def __init__(self, time, omega_3=10.0, L=50.0, max_amplitude=5.0, N=100, eps=0.01):
    #super().__init__(N)      
    Curve.__init__(self, N)  

    # 1. Define the arc length parameter 's' (from 0 to L)
    s = np.linspace(0.0, L, N)
    self.t = s
    self.lengths = s
    self.length = L
        
    # 2. Define the initial, unbent geometry (Straight rod along the X-axis)
    x = s
    y = np.zeros(N)
    z = np.zeros(N)
    self.points = np.column_stack((x, y, z))    
    
    # 3. Define the constant Normal Vectors (Vibration direction: Y-axis)
    # The vibration happens in the y-z plane perpendicular to the rod axis (x).
    # We choose the Y-axis as the bending direction.
    self.normals = np.column_stack((np.zeros(N), np.ones(N), np.zeros(N)))
        
    # --- Bending Calculation ---
    # 4. Calculate the Spatial Deflection Profile V3(s)
    V_s = V3_mode_shape(s, L=L, normalize=True)     

    # 5. Calculate the Temporal Displacement Factor (oscillation)
    temporal_factor = np.cos(omega_3 * time)    
    
    # 6. Apply the Displacement
    displacement_magnitude = max_amplitude * V_s * temporal_factor           # Shape (N,): a single magnitude for each point. 
    displacement_magnitude_reshaped = displacement_magnitude[:, np.newaxis]  # Shape (N,1): Reshape for element-wise multiplication with the 3D normals
    
    # Displacement vector = Magnitude * Normal Vector
    displacement_vectors = displacement_magnitude_reshaped * self.normals
    
    # 7. Update Points: New Position = Original Position + Displacement Vector
    self.points_bent = self.points + displacement_vectors
    self.points = self.points_bent
    
    self.calculate_tangents()
    #self.calculate_frenet_normals_from_points()
    self.calculate_Frenet_normals_from_tangents()

