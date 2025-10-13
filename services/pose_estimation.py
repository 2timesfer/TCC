import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple, Any

# --- Inicialização dos Modelos ---

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose_mediapipe = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Carrega o modelo OpenPose pré-treinado do OpenCV
# Nota: Estes arquivos .prototxt e .caffemodel precisam estar acessíveis.
# Em uma aplicação real, o caminho viria de uma configuração.
try:
    protoFile = "/content/drive/MyDrive/TCC - Debs e Fer/TCC - Dados/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "/content/drive/MyDrive/TCC - Debs e Fer/TCC - Dados/pose/mpi/pose_iter_160000.caffemodel"
    pose_openpose = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
except cv2.error as e:
    print("AVISO: Não foi possível carregar o modelo OpenPose. Verifique os caminhos dos arquivos.")
    print(f"Erro: {e}")
    pose_openpose = None

# --- Mapeamento e Padronização ---

# Mapeia os IDs do OpenPose para nomes de landmarks compreensíveis
OPENPOSE_BODY_PARTS = {
    0: "nose", 1: "neck", 2: "right_shoulder", 3: "right_elbow", 4: "right_wrist",
    5: "left_shoulder", 6: "left_elbow", 7: "left_wrist", 8: "right_hip", 9: "right_knee",
    10: "right_ankle", 11: "left_hip", 12: "left_knee", 13: "left_ankle", 14: "right_eye",
    15: "left_eye", 16: "right_ear", 17: "left_ear"
}

# Nomes dos landmarks do MediaPipe que usaremos para padronização
MEDIAPIPE_LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 
    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 
    'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 
    'right_heel', 'left_foot_index', 'right_foot_index'
]

def _map_openpose_to_standard_format(detected_points: Dict[int, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Converte a saída do OpenPose para nosso formato padrão de landmarks."""
    standard_landmarks = {}
    for part_id, coords in detected_points.items():
        part_name = OPENPOSE_BODY_PARTS.get(part_id)
        if part_name:
            standard_landmarks[part_name] = coords
    return standard_landmarks

def _map_mediapipe_to_standard_format(pose_landmarks: Any, image_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
    """Converte a saída do MediaPipe para nosso formato padrão de landmarks."""
    standard_landmarks = {}
    height, width = image_shape
    for i, landmark in enumerate(pose_landmarks.landmark):
        name = MEDIAPIPE_LANDMARK_NAMES[i]
        # Converte coordenadas normalizadas (0.0 a 1.0) para coordenadas de pixel
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        standard_landmarks[name] = (cx, cy)
    return standard_landmarks


# --- Função Pública Principal ---

def estimate_pose(frame: np.ndarray, model_name: str = 'openpose') -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Estima a pose em um único quadro de vídeo usando o modelo especificado.

    Args:
        frame (np.ndarray): O quadro do vídeo para analisar.
        model_name (str): O nome do modelo a ser usado ('openpose' ou 'mediapipe').

    Returns:
        Optional[Dict[str, Tuple[int, int]]]: 
        Um dicionário com os nomes dos landmarks e suas coordenadas (x, y),
        ou None se nenhuma pose for detectada.
    """
    if model_name == 'openpose':
        if pose_openpose is None:
            print("ERRO: Modelo OpenPose não está carregado.")
            return None
            
        frame_height, frame_width, _ = frame.shape
        in_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        
        pose_openpose.setInput(in_blob)
        output = pose_openpose.forward()

        detected_points = {}
        for i in range(len(OPENPOSE_BODY_PARTS)):
            prob_map = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            if prob > 0.1: # Limiar de confiança
                x = int((frame_width * point[0]) / output.shape[3])
                y = int((frame_height * point[1]) / output.shape[2])
                detected_points[i] = (x, y)

        return _map_openpose_to_standard_format(detected_points)

    elif model_name == 'mediapipe':
        # MediaPipe espera imagens em RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_mediapipe.process(image_rgb)
        
        if results.pose_landmarks:
            return _map_mediapipe_to_standard_format(results.pose_landmarks, frame.shape[:2])
        else:
            return None
            
    else:
        raise ValueError("Nome do modelo inválido. Escolha 'openpose' ou 'mediapipe'.")