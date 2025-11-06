import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from ergonomy_risk import process_frame_for_rula, calculate_summary_statistics

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

def process_video_and_get_rula(video_path, 
                               output_path, 
                               detector_name, 
                               user_context_dict, 
                               request_id):
    """
    Processa o vídeo, gera um novo vídeo com o esqueleto E
    gera um JSON de resumo estatístico.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return None, None

    # Configurações do vídeo de entrada
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurações do vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Lista para coletar métricas de todos os frames
    all_frame_metrics = []
    frame_count = 0

    print(f"Processando vídeo: {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        box = detect_person_box(frame)
        if box:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                all_frame_metrics.append({}) # Adiciona métrica vazia
                continue

            # Processa o ROI para análise RULA
            # **AQUI ESTÁ A MUDANÇA PRINCIPAL**
            # Agora capturamos os 3 valores de retorno
            processed_roi, rula_result, metrics = process_frame_for_rula(roi, detector_name=detector_name)
            
            # Adiciona carimbo de tempo e armazena as métricas
            metrics['timestamp_sec'] = frame_count / fps if fps > 0 else 0
            all_frame_metrics.append(metrics)
            
            # Coloca o ROI processado de volta no frame original
            frame[y1:y2, x1:x2] = processed_roi

            # Escreve o resultado RULA no frame
            if rula_result and rula_result['risk'] != 'NULL':
                text = f"RULA Score: {rula_result['score']} ({rula_result['risk']})"
                color = (0, 0, 255) if rula_result['risk'] in ['Medium risk', 'Very high risk'] else (0, 255, 0)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
             # Se nenhuma pessoa for detectada, adicione um registro vazio
             all_frame_metrics.append({'timestamp_sec': frame_count / fps if fps > 0 else 0})

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo processado e salvo em: {output_path}")

    # --- GERAÇÃO DO RESUMO PÓS-PROCESSAMENTO ---
    print("Calculando estatísticas de resumo...")
    summary_json = calculate_summary_statistics(
        all_frame_metrics,
        fps,
        total_frames,
        user_context_dict,
        request_id
    )

    return output_path, summary_json