import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

# --- Configuration & Constants ---
MP_FACE_MESH = mp.solutions.face_mesh
FACE_MESH = None 
 
def _init_face_mesh(): 
    global FACE_MESH 
    if FACE_MESH is None: 
        FACE_MESH = MP_FACE_MESH.FaceMesh( 
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5 
        )

# Quality Filter Thresholds
YAW_THRESHOLD_RATIO = 0.5  # Deviation of nose from center relative to half-width (0.0 = center, 1.0 = edge)
PITCH_THRESHOLD_RATIO = 0.5 # Similar logic for vertical
BLUR_THRESHOLD = 50.0       # Laplacian variance
MIN_FACE_SIZE = 0.1         # Face width relative to image width

# Landmark Indices (MediaPipe Topology)
# Eyes (Outer Corners)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Face Extents
CHIN = 152
FOREHEAD_ANCHOR_POINTS = [10, 151, 9] # Top of forehead area (average these)
LEFT_FACE_EDGE = 454 # Subject's Left (Image Right)
RIGHT_FACE_EDGE = 234 # Subject's Right (Image Left)

# Cheekbones (Zygo)
LEFT_CHEEKBONE = 446 # Or 352? 446 is very outer. 454 is edge. 352 is cheek. Let's use 454/234 for FW, but CW is 'Elmacık'. 
# V1 Used: RIGHT=123, LEFT=352. These are standard cheek prominences.
RIGHT_CHEEKBONE = 123
LEFT_CHEEKBONE = 352

# Jaw Corners (Gonion)
LEFT_JAW_CORNER = 58  # Right side in image (Subject's right?) No wait.
# MP: 
# 234 (Right Edge) -> ... -> 58 (Right Corner??) -> ... -> 152 (Chin)
# 454 (Left Edge) -> ... -> 288 (Left Corner??) -> ... -> 152 (Chin)
# Let's verify standard keys.
# 58 is indeed right jaw corner (Image Left).
# 288 is left jaw corner (Image Right).
RIGHT_JAW_CORNER = 58
LEFT_JAW_CORNER = 288

# Jawline Sampling (Chain from Corner to Chin)
# Right (58 to 152): 172, 136, 150, 149, 176, 148 (6 points) - V1 list verify?
# V1: [172, 136, 150, 149, 176, 148] correct.
RIGHT_JAWLINE_POINTS = [172, 136, 150, 149, 176, 148]
# Left (288 to 152): 397, 365, 379, 378, 400, 377 (6 points)
LEFT_JAWLINE_POINTS = [397, 365, 379, 378, 400, 377]

# Secondary Mandibula Points (for Angles L2/R2)
# "Jaw corner'a komşu ikinci noktalarla".
# Start from corner and go towards chin (or ear). 
# L2/R2 implies a slightly wider or narrower measurement.
# Let's use the first neighbor towards the chin for the second angle.
# Right: Corner=58. Next=172.
# Left: Corner=288. Next=397.
RIGHT_JAW_NEIGHBOR = 172
LEFT_JAW_NEIGHBOR = 397

# Side Face Points (for Side Straightness)
# Between Edge/Temple and Jaw Corner.
# Right Side (Image Left): 234 (Edge). 58 (Jaw Corner).
# Contour: 234 -> 93 -> 132 -> 58.
# Points: 93, 132. (2 internal). Maybe need more?
# Let's look slightly above 234 too? 127 is above 234.
# Segment for 'Side Straightness': From Cheek/Ear top to Jaw Corner.
# Right: 234 is prominent. 58 is corner.
# Let's take [234, 93, 132] ?
# Or include 127 (above 234).
# A longer chain gives better 'straightness' check.
# Right Side Chain: 127 -> 234 -> 93 -> 132 -> 58.
# Left Side Chain: 356 -> 454 -> 323 -> 361 -> 288.
RIGHT_SIDE_POINTS = [127, 234, 93, 132]
LEFT_SIDE_POINTS = [356, 454, 323, 361]

# Midface
NOSE_BOTTOM = 2

# --- Processing Functions ---

def get_landmarks_raw(image_path):
    """
    Reads image, checks quality, returns landmarks (pixel coords).
    Returns None if dropped.
    """
    _init_face_mesh()
    
    # 1. Read Image
    try:
        # Handle unicode paths
        file_bytes = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Read Error: {e}")
        return None

    if image is None: 
        return None
        
    h, w, _ = image.shape
    
    # 2. Blur Check
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_var < BLUR_THRESHOLD:
        # print(f"Dropped Blur ({blur_var:.1f})")
        return None # Too blurry

    # 3. Face Mesh
    results = FACE_MESH.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    # 4. Get 3D landmarks for Pose Check
    # MP returns normalized x,y,z (z is relative to head width)
    lms_3d = results.multi_face_landmarks[0].landmark
    
    # Yaw Check: Nose position relative to edges
    nose = lms_3d[1] # tip
    left_edge = lms_3d[33] # eye corner? No use edges.
    # Use 234 and 454
    r_edge = lms_3d[RIGHT_FACE_EDGE] # 234 (Image Left) - Subject Right
    l_edge = lms_3d[LEFT_FACE_EDGE] # 454 (Image Right) - Subject Left
    
    # Simple Yaw: abs(dist_nose_left - dist_nose_right) / width
    # Using X coordinates only for simple projection check
    # Note: MP x is normalized [0,1].
    
    dist_l = abs(nose.x - l_edge.x) # Nose to Image Right
    dist_r = abs(nose.x - r_edge.x) # Nose to Image Left
    width = abs(l_edge.x - r_edge.x)
    
    if width < MIN_FACE_SIZE:
        # print("Face too small")
        return None

    yaw_dev = abs(dist_l - dist_r) / width
    if yaw_dev > 0.4: # Allows some turn, but 0.5 means nose is on the edge. 0.4 is safe-ish.
        # print(f"Dropped Yaw ({yaw_dev:.2f})")
        return None
        
    # Pitch Check: Nose Y relative to Eye/Mouth center?
    # Simple aspect ratio check might catch extreme pitches too.
    # Let's skip complex pitch for now, Yaw is the main issue for shape.
    
    # Convert to pixel coords for processing
    landmarks_px = []
    for lm in lms_3d:
        landmarks_px.append([lm.x * w, lm.y * h])
        
    return np.array(landmarks_px)

def normalize_pose(landmarks):
    """
    1. Translation: Center to 0,0
    2. Roll: Rotate eyes horizontal
    3. Scale: Eye outer dist = 1.0
    """
    # Translation
    centroid = np.mean(landmarks, axis=0) # Centroid of all points
    landmarks -= centroid
    
    # Roll Correction
    # Eye Centers
    l_eye = landmarks[LEFT_EYE_OUTER] # 33
    r_eye = landmarks[RIGHT_EYE_OUTER] # 263
    
    # Vector R->L (in image: Left(263) -> Right(33))
    # Wait. 33 is Subject Left, Image Right. 263 is Subject Right, Image Left.
    # Vector from Image Left (263) to Image Right (33).
    # d = p(33) - p(263)
    d = l_eye - r_eye
    dx, dy = d[0], d[1]
    
    angle = np.arctan2(dy, dx)
    
    # Rotate
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    landmarks = np.dot(landmarks, R.T)
    
    # Scale
    # Recalc dist
    l_eye_new = landmarks[LEFT_EYE_OUTER]
    r_eye_new = landmarks[RIGHT_EYE_OUTER]
    dist = np.linalg.norm(l_eye_new - r_eye_new)
    
    scale = 1.0 / dist
    landmarks *= scale
    
    return landmarks

def extract_features(lm):
    """
    Extracts 25 metrics from normalized landmarks.
    """
    def p(idx): return lm[idx]
    def dist(i, j): return np.linalg.norm(lm[i] - lm[j])
    
    feats = {}
    
    # --- A. Basic Dimensions (8) ---
    FW = dist(RIGHT_FACE_EDGE, LEFT_FACE_EDGE)
    
    # FH: Forehead Anchor vs Chin
    # Average of 10, 151, 9
    fh_top = np.mean([p(i) for i in FOREHEAD_ANCHOR_POINTS], axis=0)
    FH = np.linalg.norm(fh_top - p(CHIN))
    
    CW = dist(RIGHT_CHEEKBONE, LEFT_CHEEKBONE)
    JW = dist(RIGHT_JAW_CORNER, LEFT_JAW_CORNER)
    
    # MFH: Eye Mid vs Nose Bottom
    eye_mid = (p(LEFT_EYE_OUTER) + p(RIGHT_EYE_OUTER)) / 2
    MFH = np.linalg.norm(eye_mid - p(NOSE_BOTTOM))
    
    LFH = dist(NOSE_BOTTOM, CHIN)
    
    R1 = FH / FW if FW > 0 else 0
    R2 = JW / CW if CW > 0 else 0
    
    feats.update({'FW': FW, 'FH': FH, 'CW': CW, 'JW': JW, 'MFH': MFH, 'LFH': LFH, 'R1': R1, 'R2': R2})
    
    # --- B. Jaw Angles (6) ---
    def get_angle(a, b, c): # Angle at b
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
    # A_L1 / A_R1 (Main Corner)
    # Right Angle (58): Ear/Edge(234) -> 58 -> Chin(152)
    A_R1 = get_angle(p(RIGHT_FACE_EDGE), p(RIGHT_JAW_CORNER), p(CHIN))
    # Left Angle (288): Ear/Edge(454) -> 288 -> Chin(152)
    A_L1 = get_angle(p(LEFT_FACE_EDGE), p(LEFT_JAW_CORNER), p(CHIN))
    
    # A_L2 / A_R2 (Neighbor)
    # Right (172): Corner(58) -> 172 -> Chin(152) (Angle inside the jawline?)
    # Or use the angle *at* 172?
    # User: "A_R2 = Sağ mandibula açısı (jaw corner’a komşu ikinci noktalarla)"
    # Likely means "Measurement of angle at the neighbor point" or "Angle using neighbor as definition".
    # Measuring convexity along the jaw.
    # Let's measure angle at 172 (formed by 58 and 136/152?).
    # Standard interpretation: The "Jaw Angle" is usually the Gonial Angle.
    # Maybe measure it slightly differently using the neighbor point to smooth out local noise.
    # Let's use: Edge(234) -> Neighbor(172) -> Chin(152). (Skips the strict corner point).
    A_R2 = get_angle(p(RIGHT_FACE_EDGE), p(RIGHT_JAW_NEIGHBOR), p(CHIN)) 
    A_L2 = get_angle(p(LEFT_FACE_EDGE), p(LEFT_JAW_NEIGHBOR), p(CHIN))
    
    A_mean = (A_L1 + A_R1 + A_L2 + A_R2) / 4.0
    A_var = np.var([A_L1, A_R1, A_L2, A_R2])
    
    feats.update({'A_L1': A_L1, 'A_R1': A_R1, 'A_L2': A_L2, 'A_R2': A_R2, 'A_mean': A_mean, 'A_var': A_var})
    
    # --- C. Jawline Deviation (6) ---
    def get_line_fit_error(point_indices):
        # Points
        pts = np.array([p(i) for i in point_indices])
        # Fit line (SVD)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        u, s, vh = np.linalg.svd(centered)
        normal = vh[-1]
        # Mean absolute distance to line
        dists = np.abs(np.dot(centered, normal))
        return np.mean(dists)

    J_dev_R = get_line_fit_error(RIGHT_JAWLINE_POINTS)
    J_dev_L = get_line_fit_error(LEFT_JAWLINE_POINTS)
    J_dev_mean = (J_dev_R + J_dev_L) / 2
    J_dev_diff = abs(J_dev_R - J_dev_L)
    
    # Chin Sharpness (Angle of segment entering chin with Vertical)
    # Vertical vector: (0, -1) [Up] or (0, 1) [Down]
    # We normalized coordinates. (0,0) is center. Eyes horizontal.
    # Vertical axis is Y.
    # We want angle of the jawline-end segment vs the Vertical axis.
    # Small angle = V shape (steep). Large angle = Flat chin.
    def get_chin_angle(last_pt_idx):
        # Vector: Chin - LastPt
        # No, Vector: LastPt -> Chin (going down)
        # Or Chin -> LastPt (going up)
        # Let's take vector v = p(LastPt) - p(Chin). This goes UP/OUT from chin.
        v = p(last_pt_idx) - p(CHIN)
        vn = v / (np.linalg.norm(v) + 1e-6)
        # Vertical Up is (0, -1) in image coords?
        # MP: y increases downwards. So Up is -y.
        vertical = np.array([0, -1]) 
        
        # Angle
        dot = np.dot(vn, vertical)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return angle

    # Last point in chain closest to chin
    # Right Chain: ... -> 148 -> Chin. Point 148.
    # Left Chain: ... -> 377 -> Chin. Point 377.
    ChinSharp_R = get_chin_angle(RIGHT_JAWLINE_POINTS[-1]) # 148
    ChinSharp_L = get_chin_angle(LEFT_JAWLINE_POINTS[-1]) # 377
    
    feats.update({'J_dev_L': J_dev_L, 'J_dev_R': J_dev_R, 'J_dev_mean': J_dev_mean, 'J_dev_diff': J_dev_diff,
                  'ChinSharp_L': ChinSharp_L, 'ChinSharp_R': ChinSharp_R})
                  
    # --- D. Ratios (2) ---
    R3 = CW / FW if FW > 0 else 0
    R4 = LFH / FH if FH > 0 else 0
    feats.update({'R3': R3, 'R4': R4})
    
    # --- E. Side Straightness (3) ---
    # Rectangular -> Straight sides. Ovale -> Curved.
    Side_dev_R = get_line_fit_error(RIGHT_SIDE_POINTS)
    Side_dev_L = get_line_fit_error(LEFT_SIDE_POINTS)
    
    feats.update({'Side_dev_L': Side_dev_L, 'Side_dev_R': Side_dev_R, 'Side_dev_mean': (Side_dev_L + Side_dev_R)/2})
    
    # Order check
    # [FW, FH, CW, JW, MFH, LFH, R1, R2, A_L1, A_R1, A_L2, A_R2, A_mean, A_var, J_dev_L, J_dev_R, J_dev_mean, J_dev_diff, ChinSharp_L, ChinSharp_R, R3, R4, Side_dev_L, Side_dev_R, Side_dev_mean]
    
    return feats

def process_dataset(root_dir, output_file):
    classes = ['ovale', 'round', 'rectangular', 'square']
    data = []
    
    print(f"Scanning {root_dir}...")
    
    for cls in classes:
        folder = os.path.join(root_dir, cls)
        # recursuve glob
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
             files.extend(glob.glob(os.path.join(folder, ext)))
        
        print(f"Class {cls}: {len(files)} images found.")
        
        for fpath in files:
            lm = get_landmarks_raw(fpath)
            if lm is None:
                # print(f"Skipped {os.path.basename(fpath)}")
                continue
                
            norm_lm = normalize_pose(lm)
            features = extract_features(norm_lm)
            features['label'] = cls
            features['filename'] = os.path.basename(fpath)
            data.append(features)
            
    df = pd.DataFrame(data)
    
    # Reorder columns to user spec (plus label/filename)
    cols = ['FW', 'FH', 'CW', 'JW', 'MFH', 'LFH', 'R1', 'R2', 
            'A_L1', 'A_R1', 'A_L2', 'A_R2', 'A_mean', 'A_var', 
            'J_dev_L', 'J_dev_R', 'J_dev_mean', 'J_dev_diff', 
            'ChinSharp_L', 'ChinSharp_R', 
            'R3', 'R4', 
            'Side_dev_L', 'Side_dev_R', 'Side_dev_mean',
            'label', 'filename']
            
    df = df[cols]
    df.to_csv(output_file, index=False)
    print(f"Done. Saved {len(df)} samples to {output_file}")


if __name__ == "__main__":
    # Assuming standard path structure
    # root = "c:/Users/burak/Desktop/face model/men"
    # But user said "v2 klasöründe". Relative path logic.
    # The images are in the parent folder of 'v2' (implied by previous context, siblings of v1).
    # "v1" is inside "men". "v2" is inside "men". Images are in "men/ovale", etc?
    # List dir showed: "ovale", "round"... at "men" level.
    # So ROOT is "c:/Users/burak/Desktop/face model/men"
    
    ROOT_DIR = r"c:/Users/burak/Desktop/face model/men"
    OUTPUT_CSV = os.path.join(ROOT_DIR, "v2", "features.csv")
    
    process_dataset(ROOT_DIR, OUTPUT_CSV)
