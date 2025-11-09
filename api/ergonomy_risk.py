"""
Módulo de Análise Ergonômica (RULA)

Responsável por:
1. Carregar os modelos de detecção de pose (YOLO, MediaPipe, OpenPose).
2. Carregar as tabelas de pontuação RULA.
3. Fornecer uma única função 'process_frame_for_rula' que recebe
   um frame, detecta a pose, calcula o RULA, E AGORA DESENHA
   geometricamente os ângulos e seus valores no frame.

CORREÇÃO v2: A função 'draw_angle_on_frame' foi reescrita para usar
produto vetorial, garantindo que o arco do ângulo interno (<180°)
seja sempre desenhado corretamente.
"""
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import datetime
import json
from ultralytics import YOLO # type: ignore 

# --- Carregamento das Tabelas RULA ---
try:
    tablea = pd.read_csv('./data/TableA.csv')
    tableb = pd.read_csv('./data/TableB.csv')
    tablec = pd.read_csv('./data/TableC.csv')
    print("Tabelas RULA carregadas com sucesso.")
except FileNotFoundError:
    print("Erro: Tabelas RULA (.csv) não encontradas.")
    tablea, tableb, tablec = None, None, None

# --- Carregamento dos Modelos de Detecção de Pose ---
# (As seções de carregamento do MediaPipe, OpenPose, YOLO permanecem inalteradas)
# 1. MediaPipe
mp_pose = mp.solutions.pose # type: ignore
pose_mediapipe = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # type: ignore

# 2. OpenPose
try:
    proto_file = r"./models/openpose/pose_deploy_linevec_faster_4_stages.prototxt"
    weights_file = r"./models/openpose/pose_iter_160000.caffemodel"
    net_openpose = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    print("Modelo OpenPose carregado com sucesso.")
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
except cv2.error:
    print("Erro: Arquivos do modelo OpenPose (.prototxt, .caffemodel) não encontrados.")
    net_openpose = None

# 3. YOLO-Pose
try:
    pose_yolo = YOLO('yolov8n-pose.pt') 
    print("Modelo YOLO-Pose carregado com sucesso.")
    YOLO_BODY_PARTS = {
        0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
        5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow",
        9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip",
        13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle"
    }
    YOLO_POSE_PAIRS = [
        ["LShoulder", "RShoulder"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["LShoulder", "LHip"],
        ["RShoulder", "RHip"], ["LHip", "RHip"], ["LHip", "LKnee"],
        ["LKnee", "LAnkle"], ["RHip", "RKnee"], ["RKnee", "RAnkle"]
    ]
except Exception as e:
    print(f"Erro ao carregar modelo YOLO-Pose: {e}.")
    pose_yolo = None

# --- Funções de Detecção de Pose ---
# (detect_pose_mediapipe, detect_pose_openpose, detect_pose_yolo permanecem inalteradas)
def detect_pose_mediapipe(frame):
    key_points_dict = {}
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_mediapipe.process(frame_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        lm_map = {'RShoulder': 12, 'RElbow': 14, 'RWrist': 16, 'RHip': 24, 'RKnee': 26,
                  'LShoulder': 11, 'LElbow': 13, 'LWrist': 15, 'LHip': 23, 'LKnee': 25,
                  'Nose': 0, 'LEye': 2, 'REye': 5, 'LEar': 7, 'REar': 8}
        
        for name, idx in lm_map.items():
            if lm[idx].visibility > 0.5:
                key_points_dict[name] = (int(lm[idx].x * w), int(lm[idx].y * h))
            else:
                key_points_dict[name] = None
    
    return frame, key_points_dict

def detect_pose_openpose(frame):
    key_points_dict = {}
    h, w, _ = frame.shape
    
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net_openpose.setInput(inp_blob)
    output = net_openpose.forward()

    points = []
    for i in range(len(BODY_PARTS) - 1):
        heat_map = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        if conf > 0.1:
            x = int((w * point[0]) / output.shape[3])
            y = int((h * point[1]) / output.shape[2])
            points.append((x, y))
        else:
            points.append(None)
    
    for part_name, part_idx in BODY_PARTS.items():
        if part_idx < len(points):
            key_points_dict[part_name] = points[part_idx]

    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]
        if key_points_dict.get(part_a) and key_points_dict.get(part_b):
            cv2.line(frame, key_points_dict[part_a], key_points_dict[part_b], (0, 255, 255), 2)
            cv2.circle(frame, key_points_dict[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, key_points_dict[part_b], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    return frame, key_points_dict

def detect_pose_yolo(frame): 
    key_points_dict = {}
    results = pose_yolo(frame, verbose=False)
    
    if results and results[0].keypoints:
        keypoints_tensor = results[0].keypoints.xy[0]
        keypoints_conf = results[0].keypoints.conf[0]

        for idx, (x, y) in enumerate(keypoints_tensor):
            part_name = YOLO_BODY_PARTS.get(idx)
            if part_name and keypoints_conf[idx] > 0.1:
                key_points_dict[part_name] = (int(x), int(y))
            elif part_name:
                key_points_dict[part_name] = None
        
        for pair in YOLO_POSE_PAIRS:
            part_a = pair[0]
            part_b = pair[1]
            if key_points_dict.get(part_a) and key_points_dict.get(part_b):
                cv2.line(frame, key_points_dict[part_a], key_points_dict[part_b], (255, 0, 0), 2)
                cv2.circle(frame, key_points_dict[part_a], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, key_points_dict[part_b], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    return frame, key_points_dict


# --- Funções de Análise RULA ---
# (calculate_angle, classify_..., rula_risk permanecem inalteradas)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def classify_upper_arm(angle):
    if angle < 20: return 1
    if angle < 50: return 2
    if angle < 90: return 3
    return 4

def classify_lower_arm(angle):
    return 1 if 80 <= angle <= 100 else 2

def classify_wrist(angle):
    if angle < 80: return 1
    if angle <= 140: return 2
    return 3

def classify_neck(angle):
    if angle <= 30: return 1
    if angle <= 45: return 2
    if angle < 55: return 3
    return 4

def classify_trunk(angle):
    if angle < 90: return 1
    if angle <= 150: return 2
    if angle <= 200: return 3
    return 4

def classify_legs(angle):
    return 1 if 60 <= angle <= 120 else 2

def rula_risk(point_score, wrist, trunk, upper_Shoulder, lower_Limb, neck,
              wrist_twist, legs, muscle_use, force_load_a,
              force_load_b, upper_body_muscle):
    rula = {'score': 'NULL', 'risk': 'NULL'}
    if all(v is not None and v != 0 for v in (wrist, trunk, upper_Shoulder, lower_Limb, neck, wrist_twist)):
        try:
            colA = f"{wrist}WT{wrist_twist}"
            valA = tablea.loc[(tablea.UpperArm == upper_Shoulder) & (tablea.LowerArm == lower_Limb), colA].values[0]
            valA += muscle_use + force_load_a
            
            colB = f"{trunk}{legs}"
            valB = tableb.loc[tableb.Neck == neck, colB].values[0]
            valB += force_load_b + upper_body_muscle
            
            a_idx, b_idx = min(valA, 8), min(valB, 7)
            valC = tablec.loc[tablec.Score == a_idx, str(b_idx)].values[0]
            
            rula['score'] = int(valC)
            if valC <= 2: rula['risk'] = 'Negligible'
            elif valC <= 4: rula['risk'] = 'Low risk'
            elif valC <= 6: rula['risk'] = 'Medium risk'
            else: rula['risk'] = 'Very high risk'
        
        except (KeyError, IndexError):
            rula['risk'] = "Erro de Cálculo (Combinação Inválida)"
            rula['score'] = 8 
            
    return rula, point_score

# --- FUNÇÃO DE DESENHO GEOMÉTRICO (REESCRITA) ---
def draw_angle_on_frame(frame, p1, p2, p3, angle_value, color=(255, 255, 0), radius=30, thickness=2, font_scale=0.6):
    """
    Desenha um arco e o valor do ângulo no frame, corrigido para
    sempre desenhar o ângulo interno (< 180).
    p2 é o vértice do ângulo.
    """
    if None in (p1, p2, p3, angle_value):
        return frame # Não desenha se os pontos ou o ângulo não forem válidos

    # Converte para array numpy para cálculos vetoriais
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    # Vetores
    v1 = p1 - p2
    v2 = p3 - p2

    # Ângulos absolutos dos vetores (em radianos)
    angle1_rad = np.arctan2(v1[1], v1[0])
    angle2_rad = np.arctan2(v2[1], v2[0])
    
    # Produto vetorial 2D para determinar a orientação (horário vs anti-horário)
    # v1[0]*v2[1] - v1[1]*v2[0]
    cross_prod = v1[0]*v2[1] - v1[1]*v2[0]

    # `angle_value` é o ângulo interno que *sempre* queremos desenhar (0-180)
    angle_value_deg = angle_value
    
    # cv2.ellipse desenha SEMPRE anti-horário (CCW)
    if cross_prod < 0: # O sweep é CCW de v1 para v2
        start_angle_deg = np.degrees(angle1_rad)
        end_angle_deg = start_angle_deg + angle_value_deg
    else: # O sweep é CW de v1 para v2, então desenhamos CCW de v2 para v1
        start_angle_deg = np.degrees(angle2_rad)
        end_angle_deg = start_angle_deg + angle_value_deg
            
    # Desenha o arco
    cv2.ellipse(frame, tuple(p2.astype(int)), (radius, radius), 0, start_angle_deg, end_angle_deg, color, thickness, cv2.LINE_AA)

    # Posição para o texto do ângulo (no bissetor do ângulo)
    text_angle_rad = np.radians(start_angle_deg + (angle_value_deg / 2))
    text_offset_x = int(p2[0] + (radius + 15) * np.cos(text_angle_rad))
    text_offset_y = int(p2[1] + (radius + 15) * np.sin(text_angle_rad))
    
    # Garante que o texto esteja legível e não fora da tela
    h, w, _ = frame.shape
    text_offset_x = max(0, min(w - 50, text_offset_x))
    text_offset_y = max(20, min(h - 10, text_offset_y))

    cv2.putText(frame, f"{angle_value:.1f}°", (text_offset_x, text_offset_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return frame

# --- Função Principal de Processamento de Frame (COM A CORREÇÃO DE TIPO DE DADO) ---
def process_frame_for_rula(frame, detector_name='mediapipe'):
    """
    Processa um único frame, despacha para o detector de pose correto,
    calcula o RULA, e agora DESENHA os ângulos geometricamente.
    """
    if tablea is None:
        return frame, {'score': 'ERROR', 'risk': 'Tabelas RULA não carregadas'}, {}

    # Passo 1: Detectar a pose
    # (key_points daqui são seguros para JSON, pois os detectores usam int())
    key_points = {}
    if detector_name == 'mediapipe':
        frame, key_points = detect_pose_mediapipe(frame)
    elif detector_name == 'openpose':
        if net_openpose is None:
            return frame, {'score': 'ERROR', 'risk': 'Modelo OpenPose não carregado'}, {}
        frame, key_points = detect_pose_openpose(frame)
    elif detector_name == 'yolo':
        if pose_yolo is None:
            return frame, {'score': 'ERROR', 'risk': 'Modelo YOLO-Pose não carregado'}, {}
        frame, key_points = detect_pose_yolo(frame)
    else:
        raise ValueError("Detector de pose desconhecido. Escolha 'mediapipe', 'openpose' ou 'yolo'.")

    rula_result = {'score': 'NULL', 'risk': 'NULL'}
    
    metrics = {
        'upper_arm_angle': None, 'lower_arm_angle': None, 'trunk_angle': None, 'neck_angle': None,
        'upper_arm_score': 0, 'lower_arm_score': 0, 'wrist_score': 0,
        'neck_score': 0, 'trunk_score': 0, 'leg_score': 0,
        'rula_score': None, 'rula_risk': 'NULL'
    }

    if not key_points:
        return frame, rula_result, metrics, key_points 

    # --- INÍCIO DA CORREÇÃO DE TIPO (para erro JSON) ---
    # (Garante que pontos derivados do NumPy sejam `int` padrão do Python)

    # Aproximação de 'Neck' (se não detectado diretamente)
    if 'Neck' not in key_points and all(key_points.get(k) for k in ['RShoulder', 'LShoulder']):
        r_shoulder = np.array(key_points['RShoulder'])
        l_shoulder = np.array(key_points['LShoulder'])
        neck_np = ((r_shoulder + l_shoulder) / 2).astype(int)
        # CORREÇÃO: Converte (np.int32, np.int32) para (int, int)
        key_points['Neck'] = (int(neck_np[0]), int(neck_np[1]))

    # --- Cálculos de Ângulo e Pontuação ---
    
    # OM BRO / BRAÇO SUPERIOR (direito)
    if all(key_points.get(k) for k in ['RHip', 'RShoulder', 'RElbow']):
        metrics['upper_arm_angle'] = calculate_angle(key_points['RHip'], key_points['RShoulder'], key_points['RElbow'])
        metrics['upper_arm_score'] = classify_upper_arm(metrics['upper_arm_angle'])
        # Passando p1, p2 (vértice), p3
        frame = draw_angle_on_frame(frame, key_points['RHip'], key_points['RShoulder'], key_points['RElbow'], metrics['upper_arm_angle'], color=(255, 255, 0)) # Amarelo
    elif all(key_points.get(k) for k in ['LHip', 'LShoulder', 'LElbow']): # Tenta o esquerdo
         metrics['upper_arm_angle'] = calculate_angle(key_points['LHip'], key_points['LShoulder'], key_points['LElbow'])
         metrics['upper_arm_score'] = classify_upper_arm(metrics['upper_arm_angle'])
         frame = draw_angle_on_frame(frame, key_points['LHip'], key_points['LShoulder'], key_points['LElbow'], metrics['upper_arm_angle'], color=(255, 255, 0)) # Amarelo
    else:
        print("Ombro: Landmarks insuficientes para cálculo do ângulo.")

    # COTOVELO / BRAÇO INFERIOR (direito)
    if all(key_points.get(k) for k in ['RShoulder', 'RElbow', 'RWrist']):
        metrics['lower_arm_angle'] = calculate_angle(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        metrics['lower_arm_score'] = classify_lower_arm(metrics['lower_arm_angle'])
        frame = draw_angle_on_frame(frame, key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'], metrics['lower_arm_angle'], color=(255, 255, 0)) # Amarelo
    elif all(key_points.get(k) for k in ['LShoulder', 'LElbow', 'LWrist']): # Tenta o esquerdo
         metrics['lower_arm_angle'] = calculate_angle(key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'])
         metrics['lower_arm_score'] = classify_lower_arm(metrics['lower_arm_angle'])
         frame = draw_angle_on_frame(frame, key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'], metrics['lower_arm_angle'], color=(255, 255, 0)) # Amarelo
    else:
        print("Cotovelo: Landmarks insuficientes para cálculo do ângulo.")

    # TRONCO
    if all(key_points.get(k) for k in ['Neck', 'LHip', 'RHip']):
        hip_mid_np = (np.array(key_points['LHip']) + np.array(key_points['RHip'])) // 2
        # CORREÇÃO: Converte (np.int32, np.int32) para (int, int)
        hip_mid = (int(hip_mid_np[0]), int(hip_mid_np[1])) 
        
        ref_point_np = key_points.get('RKnee') # Tenta pegar o joelho
        if ref_point_np is None:
            ref_point_np = tuple(hip_mid_np + (0, 50)) # Se não achar, usa um ponto abaixo (calculado com np)
        
        # CORREÇÃO: Converte o ponto de referência para (int, int), seja do joelho ou o calculado
        ref_point = (int(ref_point_np[0]), int(ref_point_np[1]))
        
        metrics['trunk_angle'] = calculate_angle(key_points['Neck'], hip_mid, ref_point)
        metrics['trunk_score'] = classify_trunk(metrics['trunk_angle'])
        frame = draw_angle_on_frame(frame, key_points['Neck'], hip_mid, ref_point, metrics['trunk_angle'], color=(0, 255, 0), radius=50) # Verde
    else:
        print("Tronco: Landmarks insuficientes para cálculo do ângulo.")

    # PESCOÇO / CABEÇA
    if all(key_points.get(k) for k in ['Nose', 'Neck', 'LHip', 'RHip']):
        hip_mid_np = (np.array(key_points['LHip']) + np.array(key_points['RHip'])) // 2
        # CORREÇÃO: Converte (np.int32, np.int32) para (int, int)
        hip_mid = (int(hip_mid_np[0]), int(hip_mid_np[1]))
        
        neck_angle_raw = calculate_angle(hip_mid, key_points['Neck'], key_points['Nose'])
        metrics['neck_angle'] = 180 - neck_angle_raw 
        metrics['neck_score'] = classify_neck(metrics['neck_angle'])
        frame = draw_angle_on_frame(frame, hip_mid, key_points['Neck'], key_points['Nose'], metrics['neck_angle'], color=(255, 0, 0), radius=25) # Azul
    else:
        print("Pescoço: Landmarks insuficientes para cálculo do ângulo.")
    
    # --- FIM DA CORREÇÃO DE TIPO ---

    metrics['wrist_score'] = 1
    metrics['leg_score'] = 1
    
    # --- Cálculo RULA Final ---
    rula_result, _ = rula_risk(
        {}, wrist=metrics['wrist_score'], trunk=metrics['trunk_score'], upper_Shoulder=metrics['upper_arm_score'],
        lower_Limb=metrics['lower_arm_score'], neck=metrics['neck_score'], wrist_twist=1, legs=metrics['leg_score'],
        muscle_use=0, force_load_a=0, force_load_b=0, upper_body_muscle=0
    )
    
    metrics['rula_score'] = rula_result.get('score')
    metrics['rula_risk'] = rula_result.get('risk')

    return frame, rula_result, metrics, key_points