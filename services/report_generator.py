import cv2
from typing import Dict, Any, Tuple

# Define as conexões do esqueleto para o desenho.
SKELETON_CONNECTIONS = [
    # Tronco
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),

    # Braço Esquerdo
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    # Braço Direito
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Perna Esquerda
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    # Perna Direita
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    
    # Cabeça
    ("nose", "neck")
]

def draw_skeleton(frame: Any, landmarks: Dict[str, Tuple[int, int]]) -> Any:
    """
    Desenha os pontos (articulações) e as linhas (ossos) do esqueleto no frame.
    """
    if not landmarks:
        return frame

    # 1. Desenha as linhas/ossos
    for point1_name, point2_name in SKELETON_CONNECTIONS:
        # Verifica se ambos os pontos da conexão existem no dicionário de landmarks
        if point1_name in landmarks and point2_name in landmarks:
            point1_coords = landmarks[point1_name]
            point2_coords = landmarks[point2_name]
            
            # Desenha a linha no frame
            cv2.line(frame, point1_coords, point2_coords, (0, 255, 0), 2) # Linha verde com 2px de espessura

    # 2. Desenha os círculos/articulações
    for landmark_name, coords in landmarks.items():
        # Desenha um círculo em cada landmark detectado
        cv2.circle(frame, coords, 4, (0, 0, 255), -1) # Círculo vermelho preenchido com 4px de raio

    return frame

def draw_rula_score(frame: Any, score: int, risk: str) -> Any:
    """
    Escreve a pontuação RULA e o nível de risco no frame.
    """
    text = f"RULA Score: {score} - {risk}"
    
    # Adiciona um fundo semi-transparente para melhor legibilidade
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (5, 5), (15 + text_width, 35 + text_height), (0, 0, 0), -1)
    
    # Escreve o texto
    cv2.putText(frame, text, (10, 30 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame