import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from ergonomy_risk import process_frame_for_rula

# Modelo YOLOv8 para detecção de pessoa
model = YOLO('yolov8n.pt')

def detect_person_box(frame, margin_pct=0.15):
    """
    Detecta a maior pessoa no frame e retorna a bounding box com margens.
    """
    res = model(frame, imgsz=640)[0]
    boxes = []
    for bbox, cls in zip(res.boxes.xyxy, res.boxes.cls):
        if int(cls) == 0:  # Classe 'pessoa'
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int).tolist()
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return None
    # Seleciona a bounding box com maior área
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    h, w = frame.shape[:2]
    mx = int((x2 - x1) * margin_pct)
    my = int((y2 - y1) * margin_pct)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(w, x2 + mx),
        min(h, y2 + my)
    )

def process_video_and_get_rula(video_path, output_path, detector_name):
    """
    Processa o vídeo, realiza a análise RULA e gera um novo vídeo com o esqueleto e o risco.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Configurações do vídeo de saída
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        box = detect_person_box(frame)
        if box:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            # Processa o ROI para análise RULA (chamada simplificada)
            processed_roi, rula_result = process_frame_for_rula(roi, detector_name=detector_name)
            
            # Coloca o ROI processado de volta no frame original
            frame[y1:y2, x1:x2] = processed_roi

            # Escreve o resultado RULA no frame
            if rula_result and rula_result['risk'] != 'NULL':
                text = f"RULA Score: {rula_result['score']} ({rula_result['risk']})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo processado e salvo em: {output_path}")
    return output_path