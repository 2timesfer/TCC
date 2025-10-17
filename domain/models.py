import cv2
from ultralytics import YOLO # type: ignore
import numpy as np

# Carrega o modelo YOLOv8 pré-treinado para detecção de objetos
model = YOLO('yolov8n.pt')

def detect_person_roi(frame: np.ndarray, margin_pct: float = 0.15) -> np.ndarray | None:
    """
    Detecta a maior pessoa em um quadro de vídeo e retorna a Região de Interesse (ROI).

    Args:
        frame (np.ndarray): O quadro de vídeo de entrada.
        margin_pct (float): A porcentagem de margem a ser adicionada ao redor da caixa delimitadora.

    Returns:
        np.ndarray | None: A imagem da ROI recortada se uma pessoa for detectada, caso contrário, None.
    """
    # Realiza a detecção de objetos no quadro
    results = model(frame, classes=0, verbose=False)[0]
    
    person_boxes = []
    # Itera sobre as detecções e filtra por 'pessoa' (classe 0)
    for bbox, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:  # Classe 'pessoa' no COCO dataset
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int).tolist()
            person_boxes.append((x1, y1, x2, y2))

    # Se nenhuma pessoa for detectada, retorna None
    if not person_boxes:
        return None

    # Encontra a maior caixa delimitadora (maior área)
    x1, y1, x2, y2 = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    # Adiciona uma margem à caixa delimitadora
    h, w = frame.shape[:2]
    margin_x = int((x2 - x1) * margin_pct)
    margin_y = int((y2 - y1) * margin_pct)

    # Calcula as novas coordenadas com margem, garantindo que estejam dentro dos limites da imagem
    return ( 
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(w, x2 + margin_x),
        min(h, y2 + margin_y)
    ) # type: ignore