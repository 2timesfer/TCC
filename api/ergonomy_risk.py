import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
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
    # Mapeamento de partes do corpo do OpenPose
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

# 3. YOLO-Pose (NOVO)
try:
    pose_yolo = YOLO('yolov8n-pose.pt') # Carrega o modelo YOLO-Pose
    print("Modelo YOLO-Pose carregado com sucesso.")
    # Mapeamento de índices do YOLO para os nomes do seu padrão
    YOLO_BODY_PARTS = {
        0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
        5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow",
        9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip",
        13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle"
    }
    # Pares de pose para desenhar o YOLO (pode ser ajustado)
    YOLO_POSE_PAIRS = [
        ["LShoulder", "RShoulder"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["LShoulder", "LHip"],
        ["RShoulder", "RHip"], ["LHip", "RHip"], ["LHip", "LKnee"],
        ["LKnee", "LAnkle"], ["RHip", "RKnee"], ["RKnee", "RAnkle"]
    ]
except Exception as e:
    print(f"Erro ao carregar modelo YOLO-Pose: {e}. Verifique se 'ultralytics' está instalado.")
    pose_yolo = None


# --- Funções de Detecção de Pose (Novas e Refatoradas) ---

def detect_pose_mediapipe(frame):
    """
    Detecta a pose usando MediaPipe e retorna o frame desenhado e os keypoints padronizados.
    """
    key_points_dict = {}
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_mediapipe.process(frame_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
        # Mapeamento de landmarks do MediaPipe para o nosso formato padrão
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
    """
    Detecta a pose usando OpenPose e retorna o frame desenhado e os keypoints padronizados.
    """
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
    
    # Mapeia os pontos detectados para o nosso dicionário padrão
    for part_name, part_idx in BODY_PARTS.items():
        if part_idx < len(points):
            key_points_dict[part_name] = points[part_idx]

    # Desenha o esqueleto
    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]
        if key_points_dict.get(part_a) and key_points_dict.get(part_b):
            cv2.line(frame, key_points_dict[part_a], key_points_dict[part_b], (0, 255, 255), 2)
            cv2.circle(frame, key_points_dict[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, key_points_dict[part_b], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    return frame, key_points_dict

def detect_pose_yolo(frame): # <--- FUNÇÃO ADICIONADA
    """
    Detecta a pose usando YOLO-Pose e retorna o frame desenhado e os keypoints padronizados.
    """
    key_points_dict = {}
    results = pose_yolo(frame, verbose=False)
    
    if results and results[0].keypoints:
        # Pega os keypoints da primeira pessoa detectada
        keypoints_tensor = results[0].keypoints.xy[0]
        keypoints_conf = results[0].keypoints.conf[0]

        for idx, (x, y) in enumerate(keypoints_tensor):
            part_name = YOLO_BODY_PARTS.get(idx)
            if part_name and keypoints_conf[idx] > 0.1:
                key_points_dict[part_name] = (int(x), int(y))
            elif part_name:
                key_points_dict[part_name] = None
        
        # Desenha o esqueleto
        for pair in YOLO_POSE_PAIRS:
            part_a = pair[0]
            part_b = pair[1]
            if key_points_dict.get(part_a) and key_points_dict.get(part_b):
                cv2.line(frame, key_points_dict[part_a], key_points_dict[part_b], (255, 0, 0), 2) # Cor Azul
                cv2.circle(frame, key_points_dict[part_a], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED) # Cor Verde
                cv2.circle(frame, key_points_dict[part_b], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED) # Cor Verde

    return frame, key_points_dict

# --- Funções de Análise RULA (O seu código) ---

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
            rula['score'] = 8 # Risco máximo por segurança
            
    return rula, point_score


# --- Função Principal de Processamento (O "Despachante") ---

def process_frame_for_rula(frame, detector_name='mediapipe'):
    """
    Processa um único frame, despachando para o detector de pose correto,
    e depois calcula o RULA.
    """
    if tablea is None:
        return frame, {'score': 'ERROR', 'risk': 'Tabelas RULA não carregadas'}

    # Passo 1: Detectar a pose com o detector escolhido
    key_points = {}
    if detector_name == 'mediapipe':
        frame, key_points = detect_pose_mediapipe(frame)
    
    elif detector_name == 'openpose':
        if net_openpose is None:
            return frame, {'score': 'ERROR', 'risk': 'Modelo OpenPose não carregado'}
        frame, key_points = detect_pose_openpose(frame)
    
    elif detector_name == 'yolo': # <--- OPÇÃO ADICIONADA
        if pose_yolo is None:
            return frame, {'score': 'ERROR', 'risk': 'Modelo YOLO-Pose não carregado'}
        frame, key_points = detect_pose_yolo(frame)
    
    else:
        raise ValueError("Detector de pose desconhecido. Escolha 'mediapipe', 'openpose' ou 'yolo'.")

    rula_result = {'score': 'NULL', 'risk': 'NULL'}
    
    # Passo 2: Calcular ângulos e RULA (lógica unificada)
    if not key_points:
        return frame, rula_result

    upper_arm_score, lower_arm_score, wrist_score = 0, 0, 0
    neck_score, trunk_score, leg_score = 0, 0, 0

    # Tenta calcular os ângulos
    # Nota: YOLO não tem 'Neck', então o cálculo de neck/trunk falhará.
    # Precisamos de uma lógica de aproximação se 'Neck' não existir
    
    if 'Neck' not in key_points and all(key_points.get(k) for k in ['RShoulder', 'LShoulder']):
        # Aproxima 'Neck' como o ponto médio dos ombros
        r_shoulder = np.array(key_points['RShoulder'])
        l_shoulder = np.array(key_points['LShoulder'])
        key_points['Neck'] = tuple(((r_shoulder + l_shoulder) / 2).astype(int))

    if all(key_points.get(k) for k in ['RShoulder', 'RElbow', 'RHip']):
        upper_arm_angle = calculate_angle(key_points['RHip'], key_points['RShoulder'], key_points['RElbow'])
        upper_arm_score = classify_upper_arm(upper_arm_angle)

    if all(key_points.get(k) for k in ['RShoulder', 'RElbow', 'RWrist']):
        lower_arm_angle = calculate_angle(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        lower_arm_score = classify_lower_arm(lower_arm_angle)
    
    if all(key_points.get(k) for k in ['Neck', 'LHip', 'RHip']):
        hip_mid = (np.array(key_points['LHip']) + np.array(key_points['RHip'])) // 2
        trunk_angle = calculate_angle(key_points['Neck'], tuple(hip_mid), key_points.get('RKnee', tuple(hip_mid + (0, 50))))
        trunk_score = classify_trunk(trunk_angle)

    if all(key_points.get(k) for k in ['Nose', 'Neck', 'LHip', 'RHip']):
        hip_mid = (np.array(key_points['LHip']) + np.array(key_points['RHip'])) // 2
        neck_angle = calculate_angle(tuple(hip_mid), key_points['Neck'], key_points['Nose'])
        neck_score = classify_neck(180 - neck_angle)


    # Simplificações mantidas
    wrist_score = 1
    leg_score = 1
    
    rula_result, _ = rula_risk(
        {}, wrist=wrist_score, trunk=trunk_score, upper_Shoulder=upper_arm_score,
        lower_Limb=lower_arm_score, neck=neck_score, wrist_twist=1, legs=leg_score,
        muscle_use=0, force_load_a=0, force_load_b=0, upper_body_muscle=0
    )

    return frame, rula_result