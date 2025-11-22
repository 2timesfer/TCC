"""
Módulo de Processamento de Vídeo (Coletor)

Responsável por:
1. Abrir um arquivo de vídeo.
2. Detectar a pessoa em cada frame (usando YOLOv8).
3. Enviar o ROI (Região de Interesse) da pessoa para 'ergonomy_risk.py'.
4. Coletar a lista de métricas brutas de todos os frames.
5. Salvar o vídeo processado (com desenhos) em um arquivo .mp4.
6. Salvar a lista completa de métricas brutas em um arquivo .json.

NOTA: O desenho dos ângulos geométricos agora é feito DIRETAMENTE em ergonomy_risk.py.

CORREÇÃO DE ORIENTAÇÃO:
Ao invés de confiar em cap.get(cv2.CAP_PROP_FRAME_WIDTH), que pode ler metadados
incorretos se o vídeo estiver rotacionado (ex: celular em pé), nós lemos o
primeiro frame real para determinar o tamanho exato da matriz de pixels.
Isso garante que o vídeo de saída tenha a mesma geometria que o OpenCV lê,
evitando vídeos esticados ou ilegíveis.
"""
import cv2
import numpy as np
import json
import datetime
import os
from ultralytics import YOLO # type: ignore

# Importa APENAS a função de processamento de frame, que agora também desenha
from ergonomy_risk import process_frame_for_rula

# Modelo YOLOv8 para detecção de pessoa
model = YOLO('yolov8n.pt')

def detect_person_box(frame, margin_pct=0.15):
    """
    Detecta a maior pessoa no frame e retorna a bounding box com margens.
    """
    res = model(frame, imgsz=640, verbose=False)[0] 
    boxes = []
    for bbox, cls in zip(res.boxes.xyxy, res.boxes.cls):
        if int(cls) == 0:  # Classe 'pessoa'
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int).tolist()
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return None
    
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    h, w = frame.shape[:2]
    mx = int((x2 - x1) * margin_pct)
    my = int((y2 - y1) * margin_pct)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(w, x2 + mx),
        min(h, y2 + my))

def process_video_and_collect_raw_data(video_path, 
                                       output_path, 
                                       detector_name,
                                       raw_data_json_path): 
    """
    Processa o vídeo, gera um novo vídeo com o esqueleto E
    salva os dados brutos de CADA frame em um JSON.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return None, None

    # --- CORREÇÃO DE ORIENTAÇÃO/DIMENSÃO ---
    # Lemos o primeiro frame para pegar as dimensões REAIS da matriz de imagem.
    # Metadados (CAP_PROP_WIDTH) mentem se houver rotação de celular.
    ret, first_frame = cap.read()
    if not ret:
        print("Não foi possível ler o primeiro frame do vídeo.")
        return None, None
    
    # As dimensões do Writer DEVEM casar com o shape do frame lido
    frame_height, frame_width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Reinicia o vídeo para o começo para processar o frame 0 novamente no loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_frame_metrics = []
    frame_count = 0

    print(f"Processando vídeo: {video_path} | Dimensões detectadas: {frame_width}x{frame_height}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        box = detect_person_box(frame)
        if box:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                all_frame_metrics.append({}) 
                out.write(frame) # Escreve o frame original se falhar ROI
                continue

            # Chama o 'ergonomy_risk'. Ele AGORA também desenha os ângulos no 'processed_roi'.
            processed_roi, rula_result, metrics, key_points = process_frame_for_rula(roi, detector_name=detector_name)
            
            # Adiciona os key_points brutos ao dicionário de métricas
            metrics['key_points'] = {k: v for k, v in key_points.items() if v is not None}

            # ARMAZENA as métricas para o JSON
            metrics['timestamp_sec'] = frame_count / fps if fps > 0 else 0
            all_frame_metrics.append(metrics)
            
            # COLOCA O ROI (agora com esqueleto E ângulos desenhados) de volta no frame
            frame[y1:y2, x1:x2] = processed_roi

            # DESENHA o score RULA DESTE FRAME
            if rula_result and rula_result['risk'] != 'NULL':
                text = f"RULA Score: {rula_result['score']} ({rula_result['risk']})"
                color = (0, 0, 255) if rula_result['risk'] in ['Medium risk', 'Very high risk'] else (0, 255, 0)
                # Verifica se o texto cabe na tela
                text_y = max(30, y1 - 10)
                cv2.putText(frame, text, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        else:
             all_frame_metrics.append({'timestamp_sec': frame_count / fps if fps > 0 else 0})

        # SALVA o frame no vídeo de saída
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo processado e salvo em: {output_path}")

    # --- SALVAR DADOS BRUTOS EM JSON ---
    try:
        with open(raw_data_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frame_metrics, f, ensure_ascii=False, indent=4)
        print(f"Dados brutos de {frame_count} frames salvos em: {raw_data_json_path}")
    except Exception as e:
        print(f"Erro ao salvar JSON de dados brutos: {e}")
        return output_path, None

    return output_path, raw_data_json_path