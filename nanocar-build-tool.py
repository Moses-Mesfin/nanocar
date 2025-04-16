import os 
import numpy as np
from tkinter import Tk, filedialog

# --------------------------------------------------
# Set working directory to the script's directory.
# --------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print("Working directory set to:", os.getcwd())


# Prompt the user for the file names/paths

# Hide the main Tkinter window
root = Tk()
root.withdraw()

motor_file = filedialog.askopenfilename(title="Select the motor XYZ file", filetypes=[("XYZ files", "*.xyz")])
axle_file = filedialog.askopenfilename(title="Select the axle XYZ file", filetypes=[("XYZ files", "*.xyz")])
wheel_file = filedialog.askopenfilename(title="Select the wheel XYZ file", filetypes=[("XYZ files", "*.xyz")])

# Adjustable parameters (units assumed in Angstrom)
WHEEL_OFFSET = 1.4          # How far to push the wheel outward from the axle attachment.
AXLE_OFFSET  = 1.4          # Additional offset along side_direction for attaching the axle.

#Can ignore these
# Axle flip flags (already used in previous steps):
FLIP_LEFT_AXLE  = True      # Reflect the left axle, if needed.
FLIP_RIGHT_AXLE = False     # Reflect the right axle, if needed.
# New wheel flip flags (not used now – wheels will be attached as desired):
FLIP_WHEELS_LEFT  = True    
FLIP_WHEELS_RIGHT = False   

# ========================
# Utility Functions
# ========================

def read_xyz(filename):
    """Read an XYZ file and return a list of (element, np.array([x, y, z]))."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+natoms]:
        parts = line.split()
        element = parts[0]
        coords = np.array(list(map(float, parts[1:4])))
        atoms.append((element, coords))
    return atoms

def write_xyz(filename, atoms, comment="Generated nanocar"):
    """Write atoms (list of (element, np.array)) to an XYZ file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for element, coords in atoms:
            f.write(f"{element} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that rotates vec1 to vec2.
    Returns the identity matrix if the vectors are parallel.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    cross = np.cross(a, b)
    dot = np.dot(a, b)
    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3)
    s = np.linalg.norm(cross)
    skew = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    R = np.eye(3) + skew + np.dot(skew, skew) * ((1 - dot) / (s**2))
    return R

def apply_transformation(molecule, R, T):
    """Apply a rotation (R) and translation (T) to each atom in the molecule."""
    new_mol = []
    for element, coord in molecule:
        new_coord = R.dot(coord) + T
        new_mol.append((element, new_coord))
    return new_mol

def get_atom(molecule, index):
    """Return the atom (element, coordinate) at the specified index."""
    return molecule[index]

def midpoint(coord1, coord2):
    """Return the midpoint between two coordinates."""
    return (coord1 + coord2) / 2.0

def reflect_across_plane(molecule, pivot, n):
    """
    Reflect each atom in 'molecule' across a plane that passes through 'pivot'
    with unit normal vector 'n'. Formula: P' = pivot + (P - pivot) - 2*((P - pivot)·n)* n.
    """
    new_mol = []
    for element, coord in molecule:
        v = coord - pivot
        new_coord = pivot + v - 2 * np.dot(v, n) * n
        new_mol.append((element, new_coord))
    return new_mol

def rotate_180_about_axis(molecule, axis, pivot):
    """
    Rotate the molecule 180° about a given axis (unit vector) passing through pivot.
    The rotation matrix for 180° about axis u is: R = 2*outer(u,u) - I.
    """
    R = 2 * np.outer(axis, axis) - np.eye(3)
    return apply_transformation(molecule, R, T = pivot - R.dot(pivot))

# ========================
# Assembly Functions
# ========================

def orient_axle(axle, target_direction):
    """
    Rotate the axle so that its wheel-connector vector (from index 0 to 1)
    is reoriented so that its projection lies in the plane perpendicular to target_direction.
    Returns the rotated axle molecule and the normalized projected vector.
    """
    # Use axle connector atoms at indices 0 and 1 (for wheels).
    a1 = get_atom(axle, 0)[1]
    a2 = get_atom(axle, 1)[1]
    axle_vector = a2 - a1

    target_unit = target_direction / np.linalg.norm(target_direction)
    proj = axle_vector - np.dot(axle_vector, target_unit) * target_unit
    if np.linalg.norm(proj) < 1e-8:
        desired = axle_vector
    else:
        desired = proj / np.linalg.norm(proj)
    
    R = rotation_matrix_from_vectors(axle_vector, desired)
    rotated_axle = apply_transformation(axle, R, T=np.zeros(3))
    return rotated_axle, desired

def align_fragment(mol, connector_index, target_position, desired_direction, connector_pair_index=None):
    """
    Rotate and translate a fragment so that:
      - (If provided) the vector defined by atoms at connector_index and connector_pair_index
        aligns with desired_direction.
      - Then translate so that the atom at connector_index reaches target_position.
    This function is used for aligning wheels.
    """
    if connector_pair_index is not None:
        conn_coord = get_atom(mol, connector_index)[1]
        pair_coord = get_atom(mol, connector_pair_index)[1]
        frag_vector = pair_coord - conn_coord
        R = rotation_matrix_from_vectors(frag_vector, desired_direction)
    else:
        R = np.eye(3)
    
    rotated = apply_transformation(mol, R, T=np.zeros(3))
    connector_pos = get_atom(rotated, connector_index)[1]
    T = target_position - connector_pos
    aligned = apply_transformation(rotated, np.eye(3), T)
    return aligned

# ========================
# Main Assembly Process
# ========================

def build_nanocar():
    # Read parts.
    motor = read_xyz(motor_file)
    axle_template = read_xyz(axle_file)
    wheel_template = read_xyz(wheel_file)
    
    # --- Motor ---
    # Assume the motor has its connector atoms at indices 0 and 1.
    motor_conn_left  = get_atom(motor, 0)[1]
    motor_conn_right = get_atom(motor, 1)[1]
    # The third motor atom (index 2) helps define the motor’s plane.
    
    # Compute the motor center and derive side directions.
    motor_center = (motor_conn_left + motor_conn_right) / 2.0
    left_direction = motor_conn_left - motor_center
    left_direction /= np.linalg.norm(left_direction)
    right_direction = motor_conn_right - motor_center
    right_direction /= np.linalg.norm(right_direction)
    
    final_atoms = motor.copy()
    
    # --- Axles ---
    # Each axle has three connector atoms:
    #   - Indices 0 & 1: wheel-attachment points.
    #   - Index 2: motor-attachment point.
    # (Leave this section unchanged.)
    axles_data = []
    for side_label, motor_conn, side_direction in [
            ("left", motor_conn_left, left_direction),
            ("right", motor_conn_right, right_direction)]:
        
        axle = [(el, np.copy(coord)) for el, coord in axle_template]
        # Orient the axle using side_direction.
        axle_oriented, _ = orient_axle(axle, side_direction)
        
        # Apply an offset along side_direction.
        desired_motor_conn = motor_conn + AXLE_OFFSET * side_direction
        axle_motor_connector = get_atom(axle_oriented, 2)[1]
        T = desired_motor_conn - axle_motor_connector
        axle_aligned = apply_transformation(axle_oriented, np.eye(3), T)
        
        # Optionally flip (via reflection) the axle.
        if (side_label == "left" and FLIP_LEFT_AXLE) or (side_label == "right" and FLIP_RIGHT_AXLE):
            pivot = get_atom(axle_aligned, 2)[1]  # Pivot at the motor connector.
            axle_aligned = reflect_across_plane(axle_aligned, pivot, side_direction)
        
        axles_data.append((axle_aligned, side_label))
        final_atoms.extend(axle_aligned)
    
    # --- Wheels ---
    # Here, we attach one wheel for each axle attachment point individually.
    assembled_wheels = []
    for axle_aligned, side_label in axles_data:
        # Compute the positions of the two wheel-attachment atoms.
        pos0 = get_atom(axle_aligned, 0)[1]
        pos1 = get_atom(axle_aligned, 1)[1]
        axle_center = (pos0 + pos1) / 2.0
        
        # Attach the wheel at attachment point 0.
        dir0 = pos0 - axle_center
        norm0 = np.linalg.norm(dir0)
        if norm0 < 1e-8:
            dir0 = np.array([1.0, 0.0, 0.0])
        else:
            dir0 /= norm0
        wheel_target_position0 = pos0 + WHEEL_OFFSET * dir0
        wheel0 = [(el, np.copy(coord)) for el, coord in wheel_template]
        # Use the default wheel attachment: connector index 0, partner index 1.
        wheel_aligned0 = align_fragment(wheel0,
                                        connector_index=0,
                                        target_position=wheel_target_position0,
                                        desired_direction=dir0,
                                        connector_pair_index=1)
        
        # Attach the wheel at attachment point 1.
        dir1 = pos1 - axle_center
        norm1 = np.linalg.norm(dir1)
        if norm1 < 1e-8:
            dir1 = np.array([1.0, 0.0, 0.0])
        else:
            dir1 /= norm1
        wheel_target_position1 = pos1 + WHEEL_OFFSET * dir1
        wheel1 = [(el, np.copy(coord)) for el, coord in wheel_template]
        # Use the default wheel attachment: connector index 0, partner index 1.
        wheel_aligned1 = align_fragment(wheel1,
                                        connector_index=0,
                                        target_position=wheel_target_position1,
                                        desired_direction=dir1,
                                        connector_pair_index=1)
        
        assembled_wheels.extend(wheel_aligned0)
        assembled_wheels.extend(wheel_aligned1)
    
    final_atoms.extend(assembled_wheels)
    
    output_file = "nanocar.xyz"
    write_xyz(output_file, final_atoms, comment="Nanocar assembled from motor, axles, and wheels")
    print(f"Nanocar model written to {output_file}")

if __name__ == "__main__":
    build_nanocar()
