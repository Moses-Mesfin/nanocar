import os
import numpy as np
from tkinter import Tk, filedialog
from copy import deepcopy

# Working directory
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# Prompt for file selection
root = Tk()
root.withdraw()

###
motor_file = filedialog.askopenfilename(title="Select the motor XYZ file")
axle_file = filedialog.askopenfilename(title="Select the axle XYZ file")
wheel_file = filedialog.askopenfilename(title="Select the wheel XYZ file")


# Adjustable parameters (units assumed in Angstrom)
WHEEL_OFFSET = 1.4          # How far to push the wheel outward from the axle attachment.
AXLE_OFFSET  = 1.4          # Additional offset along side_direction for attaching the axle.

#Can ignore these
# Axle flip flags (already used in previous steps):
FLIP_LEFT_AXLE  = True      # Reflect the left axle, if needed.
FLIP_RIGHT_AXLE = False     # Reflect the right axle, if needed.



# Trajectory parameters  ###
fixed_atoms_count = int(input("How many atoms at the start of the motor should remain fixed? (23 for current testing example)"))
n_frames = int(input("Number of frames for trajectory (1800 recommended): "))
angle_per_frame = float(input("Rotation angle per frame (1-5 degree recommended): "))

#fixed_atoms_count = 23
#n_frames = 1800
#angle_per_frame = 1


# Helper functions (read_xyz, write_xyz_frame, rotation_matrix, etc.)


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


def build_cu_surface(nx=10, ny=10, spacing=2.55, z_height=0.0):
    """
    Build a flat Cu(111)-like surface as a simple square grid.
    nx, ny = number of atoms along x and y
    spacing = distance between atoms (approx nearest neighbor spacing in Å)
    z_height = z-coordinate of the surface
    """
    atoms = []
    for i in range(nx):
        for j in range(ny):
            x = i * spacing
            y = j * spacing
            z = z_height
            atoms.append(("Cu", np.array([x, y, z])))
    return atoms


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

def build_nanocar(motor_file, axle_file, wheel_file):
    motor = read_xyz(motor_file)
    axle_template = read_xyz(axle_file)
    wheel_template = read_xyz(wheel_file)
    
    motor_conn_left  = get_atom(motor, 0)[1]
    motor_conn_right = get_atom(motor, 1)[1]
    motor_center = (motor_conn_left + motor_conn_right) / 2.0
    left_direction = motor_conn_left - motor_center
    left_direction /= np.linalg.norm(left_direction)
    right_direction = motor_conn_right - motor_center
    right_direction /= np.linalg.norm(right_direction)
    
    final_atoms = motor.copy()
    axles_data = []
    for side_label, motor_conn, side_direction in [
            ("left", motor_conn_left, left_direction),
            ("right", motor_conn_right, right_direction)]:
        axle = [(el, np.copy(coord)) for el, coord in axle_template]
        axle_oriented, _ = orient_axle(axle, side_direction)
        desired_motor_conn = motor_conn + AXLE_OFFSET * side_direction
        axle_motor_connector = get_atom(axle_oriented, 2)[1]
        T = desired_motor_conn - axle_motor_connector
        axle_aligned = apply_transformation(axle_oriented, np.eye(3), T)
        if (side_label == "left" and FLIP_LEFT_AXLE) or (side_label == "right" and FLIP_RIGHT_AXLE):
            pivot = get_atom(axle_aligned, 2)[1]
            axle_aligned = reflect_across_plane(axle_aligned, pivot, side_direction)
        axles_data.append((axle_aligned, side_label))
        final_atoms.extend(axle_aligned)
    
    assembled_wheels = []
    for axle_aligned, side_label in axles_data:
        pos0 = get_atom(axle_aligned, 0)[1]
        pos1 = get_atom(axle_aligned, 1)[1]
        axle_center = (pos0 + pos1) / 2.0
        dir0 = pos0 - axle_center
        dir0 /= np.linalg.norm(dir0) if np.linalg.norm(dir0) >= 1e-8 else np.array([1.0, 0.0, 0.0])
        wheel_target_position0 = pos0 + WHEEL_OFFSET * dir0
        wheel0 = [(el, np.copy(coord)) for el, coord in wheel_template]
        wheel_aligned0 = align_fragment(wheel0, 0, wheel_target_position0, dir0, connector_pair_index=1)
        dir1 = pos1 - axle_center
        dir1 /= np.linalg.norm(dir1) if np.linalg.norm(dir1) >= 1e-8 else np.array([1.0, 0.0, 0.0])
        wheel_target_position1 = pos1 + WHEEL_OFFSET * dir1
        wheel1 = [(el, np.copy(coord)) for el, coord in wheel_template]
        wheel_aligned1 = align_fragment(wheel1, 0, wheel_target_position1, dir1, connector_pair_index=1)
        assembled_wheels.extend(wheel_aligned0)
        assembled_wheels.extend(wheel_aligned1)
    
    final_atoms.extend(assembled_wheels)
    
    output_file = "nanocar.xyz"
    write_xyz(output_file, final_atoms, comment="Nanocar assembled from motor, axles, and wheels")
    print(f"Nanocar model written to {output_file}")
    return final_atoms, motor_conn_left, motor_conn_right



# ========================
# Main Assembly Process
# ========================


###


def rotation_matrix(axis, angle_deg):
    """Return the rotation matrix for rotation around 'axis' by 'angle_deg'."""
    angle_rad = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = cos_a * np.eye(3) + sin_a * np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ]) + (1 - cos_a) * np.outer(axis, axis)
    return R

def apply_rotation(atoms, axis_point, axis_dir, angle_deg):
    """Rotate atoms around axis_dir passing through axis_point by angle_deg."""
    R = rotation_matrix(axis_dir, angle_deg)
    return [(el, R @ (coord - axis_point) + axis_point) for el, coord in atoms]

def apply_translation(atoms, T):
    """Translate all atoms by translation vector T."""
    return [(el, coord + T) for el, coord in atoms]

def write_xyz_frame(f, atoms, comment="Frame"):
    """Write a single frame to XYZ file f with a comment."""
    f.write(f"{len(atoms)}\n{comment}\n")
    for el, coord in atoms:
        f.write(f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


###

# Build the initial nanocar

#motor_file = r"C:\Python\Computational_Chemistry\3_nanocar_build_tool\motor.xyz"
#axle_file = r"C:\Python\Computational_Chemistry\3_nanocar_build_tool\axle.xyz"
#wheel_file = r"C:\Python\Computational_Chemistry\3_nanocar_build_tool\wheel.xyz"

final_atoms, motor_conn_left, motor_conn_right = build_nanocar(motor_file, axle_file, wheel_file)

surface_start = len(final_atoms)


# Identify motor atoms
motor_atoms = read_xyz(motor_file)
axis_atom1 = motor_atoms[3][1]
print(f"axis_atom1 is {axis_atom1}")
axis_atom2 = motor_atoms[4][1]
print(f"axis_atom2 is {axis_atom2}")
rotation_axis = axis_atom2 - axis_atom1
rotation_axis /= np.linalg.norm(rotation_axis)

motor_fixed = motor_atoms[:fixed_atoms_count]
motor_rotating = motor_atoms[fixed_atoms_count:]


# --- Wheels & Axles Indices (computed BEFORE surface atoms) ---

motor_atoms = read_xyz(motor_file)
motor_len = len(motor_atoms)

wheel_size = len(read_xyz(wheel_file))
axle_atom_count = len(read_xyz(axle_file))  # no hardcoding

# Wheels were the last thing added in build_nanocar(), so compute start index
# relative to the nanocar BEFORE we append the Cu surface
surface_start = len(final_atoms)  # bookmark before adding surface
first_wheel_start = surface_start - 4 * wheel_size

# Build wheel index list
wheels_indices = [
    (first_wheel_start + k * wheel_size, first_wheel_start + (k + 1) * wheel_size)
    for k in range(4)
]

# Build axle indices (2 axles, right after motor)
axle_indices = [motor_len + i * axle_atom_count for i in range(2)]








# --- Auto-level the car before appending surface ---

# Step 1: Compute forward vector (same as before)
forward = motor_conn_right - motor_conn_left
forward /= np.linalg.norm(forward)

# Step 2: Estimate current "up" vector from wheels/axles
axle_coords = []
for base_idx in axle_indices:
    axle_coords.extend([final_atoms[base_idx + j][1] for j in range(axle_atom_count)])
p0, p1, p2 = axle_coords[:3]
v1 = p1 - p0
v2 = p2 - p0
current_car_normal = np.cross(v1, v2)
current_car_normal /= np.linalg.norm(current_car_normal)

# Step 3: Build orthonormal frame using forward and desired up
approx_up = np.array([0.0, 0.0, 1.0])
if abs(np.dot(forward, approx_up)) > 0.9:
    approx_up = np.array([1.0, 0.0, 0.0])
side = np.cross(forward, approx_up)
side /= np.linalg.norm(side)
desired_up = np.cross(side, forward)
desired_up /= np.linalg.norm(desired_up)

# Step 4: Compute rotation that maps current normal to desired_up
R_level = rotation_matrix_from_vectors(current_car_normal, desired_up)

# Step 5: Rotate *all* nanocar atoms
final_atoms = apply_transformation(final_atoms, R_level, T=np.zeros(3))

# Also rotate motor connectors so movement direction stays consistent
motor_conn_left = R_level @ motor_conn_left
motor_conn_right = R_level @ motor_conn_right


# Rotate motor atoms so they are leveled with the chassis too
motor_atoms = apply_transformation(motor_atoms, R_level, T=np.zeros(3))

# Recompute motor_fixed and motor_rotating in the leveled frame
motor_fixed = motor_atoms[:fixed_atoms_count]
motor_rotating = motor_atoms[fixed_atoms_count:]


# --- Small Separation Offset to Avoid Clashes ---
# Compute approximate center of rotating fragment
rotating_center = np.mean([coord for _, coord in motor_rotating], axis=0)

# Translate rotating fragment slightly along rotation axis
# Positive direction pushes it away from motor_fixed along the axis
tiny_gap = 0.3  # Å, adjust if still too close
motor_rotating = apply_translation(motor_rotating, tiny_gap * rotation_axis)

print(f"Applied small separation offset of {tiny_gap} Å along rotation axis.")


# Recompute rotation axis points in leveled frame
axis_atom1 = motor_atoms[3][1]
axis_atom2 = motor_atoms[4][1]
rotation_axis = axis_atom2 - axis_atom1
rotation_axis /= np.linalg.norm(rotation_axis)






# --- Build Cu surface aligned with car forward direction ---

spacing = 2.55
forward = motor_conn_right - motor_conn_left
forward /= np.linalg.norm(forward)

approx_up = np.array([0.0, 0.0, 1.0])
if abs(np.dot(forward, approx_up)) > 0.9:
    approx_up = np.array([1.0, 0.0, 0.0])

side = np.cross(forward, approx_up)
side /= np.linalg.norm(side)
corrected_up = np.cross(side, forward)
corrected_up /= np.linalg.norm(corrected_up)

# --- 1. Compute car bounding box in forward/side coordinates ---
car_coords = np.array([coord for _, coord in final_atoms])  # all nanocar atoms
forward_proj = car_coords @ forward        # projection along forward axis
side_proj = car_coords @ side              # projection along side axis

car_forward_min, car_forward_max = np.min(forward_proj), np.max(forward_proj)
car_side_min, car_side_max = np.min(side_proj), np.max(side_proj)
car_length = car_forward_max - car_forward_min
car_width = car_side_max - car_side_min

# --- 2. Build slightly cropped & extended Cu surface ---
# Add margins (Å) around car
forward_margin_back = spacing * 2   # space behind car
forward_margin_front = spacing * 20 # space in front of car
side_margin = spacing * 4           # side margin

forward_min = car_forward_min - forward_margin_back
forward_max = car_forward_max + forward_margin_front
side_min = car_side_min - side_margin
side_max = car_side_max + side_margin

# Determine grid size
nx = int(np.ceil((forward_max - forward_min) / spacing))
ny = int(np.ceil((side_max - side_min) / spacing))

cu_surface_aligned = []
for i in range(nx):
    for j in range(ny):
        pos = (forward_min + i * spacing) * forward + (side_min + j * spacing) * side
        cu_surface_aligned.append(("Cu", pos))

# --- 3. Center surface along side direction (optional) ---
# (This keeps car centered left-right)
surface_center_side = np.mean([np.dot(coord, side) for _, coord in cu_surface_aligned])
center_shift = ((car_side_min + car_side_max)/2 - surface_center_side) * side
cu_surface_aligned = apply_transformation(cu_surface_aligned, np.eye(3), T=center_shift)

# --- 4. Adjust height (same as before) ---
wheel_connectors = [final_atoms[start][1] for start, _ in wheels_indices]
projections = [np.dot(coord, corrected_up) for coord in wheel_connectors]
lowest_proj = min(projections)
current_plane_proj = np.dot(cu_surface_aligned[0][1], corrected_up)

clearance = 6.0  # Å
shift_distance = (lowest_proj - current_plane_proj) - clearance
cu_surface_aligned = apply_transformation(
    cu_surface_aligned,
    np.eye(3),
    T=shift_distance * corrected_up
)


# --- 5. Append to final_atoms ---
surface_start = len(final_atoms)
final_atoms = final_atoms + cu_surface_aligned

# --- Wheel geometry for rolling without slip (precompute radius & sign) ---
wheel_radii = []
wheel_roll_sign = []  # +1 or -1 per wheel

for (start, end) in wheels_indices:
    # Axis from the first two atoms of the wheel (as you already use)
    axis_point0 = final_atoms[start][1]
    axis_dir0 = final_atoms[start+1][1] - axis_point0
    axis_dir0 /= np.linalg.norm(axis_dir0)

    # Estimate radius from remaining atoms' distance to axis line
    coords = [coord for _, coord in final_atoms[start+2:end]]
    dists = []
    for p in coords:
        v = p - axis_point0
        d = np.linalg.norm(v - np.dot(v, axis_dir0) * axis_dir0)
        dists.append(d)
    radius = float(np.mean(dists)) if dists else 1.0
    wheel_radii.append(radius)

    # Rolling direction sign so translation along 'forward' spins the wheel correctly
    sgn = np.sign(np.dot(np.cross(axis_dir0, corrected_up), forward))
    wheel_roll_sign.append(1.0 if sgn == 0 else sgn)









output_file = "nanocar_trajectory_1.xyz"









# ============================
# Adjustable Motion Parameters
# ============================

#initial_motor_angle = float(input("Enter the initial motor angle (e.g., 270): "))
#start_angle = float(input("Enter motor angle where car starts moving (e.g., 0): "))
#end_angle = float(input("Enter motor angle where car stops moving (e.g., 130): "))

initial_motor_angle = 270.0   # starting motor angle (degrees)
start_angle = 0.0             # car moves starting at this relative angle (deg)
end_angle = 130.0             # car stops moving at this relative angle (deg)

# Define how far the car should move per full 360° motor rotation (Å)
distance_per_rotation = 10.0  # adjust to match your physical expectation

prev_angle = initial_motor_angle


# --- Rolling state (cumulative wheel angles, degrees) ---
wheel_spin_deg = [0.0] * len(wheels_indices)

# Visualization multiplier (tune to taste; 1.0 = physically correct; you used 10)
spin_factor = 1.0


with open(output_file, 'w') as f:
    total_forward_T = np.zeros(3)
    movement_direction = motor_conn_right - motor_conn_left
    movement_direction /= np.linalg.norm(movement_direction)

    print("wheels_indices:", wheels_indices)

    for frame in range(n_frames):
        # ---- Motor Rotation ----
        angle = initial_motor_angle + frame * angle_per_frame
        delta_angle = angle - prev_angle
        relative_angle = (angle - initial_motor_angle) % 360

        rotated_part = apply_rotation(motor_rotating, axis_atom1, rotation_axis, angle)
        motor_frame = motor_fixed + rotated_part

        frame_atoms = deepcopy(final_atoms)
        frame_atoms[:len(motor_frame)] = motor_frame

        # ======================
        # Translation Condition
        # ======================
        in_window = False
        if start_angle <= end_angle:
            in_window = start_angle <= relative_angle <= end_angle
        else:
            # Wrap-around window (e.g. start=300, end=50)
            in_window = (relative_angle >= start_angle) or (relative_angle <= end_angle)

        forward_T = np.zeros(3)
        if in_window:
            if start_angle <= end_angle:
                progress = (relative_angle - start_angle) / (end_angle - start_angle)
            else:
                total_range = (360 - start_angle) + end_angle
                if relative_angle >= start_angle:
                    progress = (relative_angle - start_angle) / total_range
                else:
                    progress = ((360 - start_angle) + relative_angle) / total_range

            # Smooth acceleration/deceleration profile (sin-shaped)
            speed_factor = np.sin(progress * np.pi)

            # Scale by actual angle rotated this frame so total distance stays constant
            distance_this_frame = (abs(delta_angle) / 360.0) * distance_per_rotation * speed_factor
            forward_T = distance_this_frame * movement_direction


        total_forward_T += forward_T

        # ----------------------
        # Wheel Rolling Rotation (cumulative)
        # ----------------------
        d_move = np.linalg.norm(forward_T)  # Å moved this frame

        # (A) Update cumulative spin only if we actually moved (prevents kickback at edges)
        if d_move > 1e-6:
            # Uncomment to debug:
            # print(f"Frame {frame}: Car moved {d_move:.4f} Å")
            for i in range(len(wheels_indices)):
                # Δθ_this_frame (deg) added to cumulative spin
                wheel_spin_deg[i] += spin_factor * (d_move / wheel_radii[i]) * (180.0 / np.pi) * wheel_roll_sign[i]

        # (B) Apply the current cumulative angle to each wheel (even if no movement this frame)
        for i, (start, end) in enumerate(wheels_indices):
            axis_point = frame_atoms[start][1]
            axis_dir = frame_atoms[start+1][1] - axis_point
            axis_dir /= np.linalg.norm(axis_dir)

            angle_now = wheel_spin_deg[i]  # total accumulated angle (deg)
            wheel_atoms = frame_atoms[start:end]
            rotated_wheel = apply_rotation(wheel_atoms, axis_point, axis_dir, angle_now)
            frame_atoms[start:end] = rotated_wheel


        # ----------------------
        # Translate + Write Frame
        # ----------------------
        nanocar_atoms = frame_atoms[:surface_start]
        surface_atoms = frame_atoms[surface_start:]
        nanocar_atoms = apply_translation(nanocar_atoms, total_forward_T)
        frame_atoms = nanocar_atoms + surface_atoms

        write_xyz_frame(f, frame_atoms, comment=f"Frame {frame}")
        prev_angle = angle







print(f"Trajectory written to {output_file}")
