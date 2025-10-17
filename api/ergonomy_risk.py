import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# --- Carregamento das Tabelas RULA ---
# As tabelas são carregadas uma vez quando o módulo é importado.
try:
    TABLE_A_PATH = './Data/TableA.csv'
    TABLE_B_PATH = './Data/TableB.csv'
    TABLE_C_PATH = './Data/TableC.csv'
    
    tablea = pd.read_csv(TABLE_A_PATH)
    tableb = pd.read_csv(TABLE_B_PATH)
    tablec = pd.read_csv(TABLE_C_PATH)
    print("Tabelas RULA carregadas com sucesso pelo módulo ergonomy_risk.")

except FileNotFoundError as e:
    print(f"Erro ao carregar tabelas RULA: {e}")
    print("Certifique-se de que os arquivos .csv estão no mesmo diretório.")
    # Define como None para que a verificação posterior falhe de forma controlada
    tablea, tableb, tablec = None, None, None

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose  # type: ignore
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # type: ignore

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

# Funções de classificação para o RULA
def classify_upper_arm(angle):
    if angle < 20:
        return 1
    if angle < 50:
        return 2
    if angle < 90:
        return 3
    return 4

def classify_lower_arm(angle):
    return 1 if 80 <= angle <= 100 else 2

def classify_wrist(angle):
    if angle < 80:
        return 1
    if angle <= 140:
        return 2
    return 3

def classify_neck(angle):
    if angle <= 30:
        return 1
    if angle <= 45:
        return 2
    if angle < 55: 
        return 3
    return 4

def classify_trunk(angle):
    if angle < 90:
        return 1
    if angle <= 150:
        return 2
    if angle <= 200:
        return 3
    return 4

def classify_legs(angle):
    return 1 if 60 <= angle <= 120 else 2
    
def get_body_key_points(landmarks, h, w):
    """Extrai as coordenadas dos landmarks de interesse."""
    points = {}
    lm_indices = {
        'LShoulder': 11, 'RShoulder': 12, 'LElbow': 13, 'RElbow': 14,
        'LWrist': 15, 'RWrist': 16, 'LHip': 23, 'RHip': 24, 'LKnee': 25, 'RKnee': 26
    }
    for name, idx in lm_indices.items():
        lm = landmarks[idx]
        if lm.visibility > 0.5:
            points[name] = (int(lm.x * w), int(lm.y * h))
        else:
            points[name] = None
    return points


def rula_risk(point_score, wrist, trunk, upper_Shoulder, lower_Limb, neck,
              wrist_twist, legs, muscle_use, force_load_a,
              force_load_b, upper_body_muscle):
    """Calcula o risco RULA com base nos scores."""
    rula = {'score': 'NULL', 'risk': 'NULL'}
    if all(v != 0 for v in (wrist, trunk, upper_Shoulder, lower_Limb, neck, wrist_twist)):
        # Tabela A
        colA = f"{wrist}WT{wrist_twist}"
        valA = tablea.loc[ # type: ignore
            (tablea.UpperArm == upper_Shoulder) & # type: ignore
            (tablea.LowerArm == lower_Limb), # type: ignore
            colA
        ].values[0] # type: ignore
        point_score['posture_score_a'] = valA
        valA += muscle_use + force_load_a
        point_score['wrist_and_arm_score'] = valA

        # Tabela B
        colB = f"{trunk}{legs}"
        valB = tableb.loc[tableb.Neck == neck, colB].values[0] # type: ignore
        point_score['posture_score_b'] = valB
        valB += force_load_b + upper_body_muscle
        point_score['neck_trunk_leg_score'] = valB

        # Tabela C
        a_idx = min(valA, 8)
        b_idx = min(valB, 7)
        valC = tablec.loc[tablec.Score == a_idx, str(b_idx)].values[0] # type: ignore

        rula['score'] = int(valC) # type: ignore
        if valC <= 2:
            rula['risk'] = 'Negligible'
        elif valC <= 4:
            rula['risk'] = 'Low risk'
        elif valC <= 6:
            rula['risk'] = 'Medium risk'
        else:
            rula['risk'] = 'Very high risk'
    return rula, point_score


def process_frame_for_rula(frame):
    """
    Processa um único frame para extrair landmarks e calcular o RULA.
    """
    if tablea is None or tableb is None or tablec is None:
        # Se as tabelas não foram carregadas, não prossiga.
        return frame, {'score': 'ERROR', 'risk': 'Tabelas não carregadas'}

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    rula_result = {'score': 'NULL', 'risk': 'NULL'}

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        key_points = get_body_key_points(landmarks, h, w)

        # Scores iniciais
        upper_arm_score, lower_arm_score, wrist_score = 0, 0, 0
        neck_score, trunk_score, leg_score = 0, 0, 0

        # Lado direito
        if all(key_points[k] for k in ['RShoulder', 'RElbow', 'RHip']):
            upper_arm_angle = calculate_angle(key_points['RHip'], key_points['RShoulder'], key_points['RElbow'])
            upper_arm_score = classify_upper_arm(upper_arm_angle)

        if all(key_points[k] for k in ['RShoulder', 'RElbow', 'RWrist']):
            lower_arm_angle = calculate_angle(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
            lower_arm_score = classify_lower_arm(lower_arm_angle)
        
        wrist_score = 1
        
        if all(key_points[k] for k in ['RShoulder', 'LShoulder', 'RHip', 'LHip']):
            shoulder_mid = ((key_points['RShoulder'][0] + key_points['LShoulder'][0]) // 2, (key_points['RShoulder'][1] + key_points['LShoulder'][1]) // 2)
            hip_mid = ((key_points['RHip'][0] + key_points['LHip'][0]) // 2, (key_points['RHip'][1] + key_points['LHip'][1]) // 2)
            neck_approx = (shoulder_mid[0], shoulder_mid[1] - 50)
            neck_angle = calculate_angle(hip_mid, shoulder_mid, neck_approx)
            neck_score = classify_neck(180 - neck_angle)
            trunk_angle = calculate_angle(neck_approx, hip_mid, key_points['RKnee']) if key_points['RKnee'] else 90
            trunk_score = classify_trunk(trunk_angle)

        leg_score = 1
        
        point_score = {}
        rula_result, _ = rula_risk(
            point_score,
            wrist=wrist_score, trunk=trunk_score, upper_Shoulder=upper_arm_score,
            lower_Limb=lower_arm_score, neck=neck_score,
            wrist_twist=1, legs=leg_score, muscle_use=0, force_load_a=0,
            force_load_b=0, upper_body_muscle=0
        )

    return frame, rula_result