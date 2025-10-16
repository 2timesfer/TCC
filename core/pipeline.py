import cv2
import imageio.v2 as imageio
import numpy as np
from typing import Dict, Any, Optional, Tuple

from domain import rula_calculator
from services import pose_estimation
from services import report_generator

def calculate_scores_from_landmarks(landmarks: Dict[str, Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    """
    Calcula todos os ângulos e os converte em scores RULA a partir dos landmarks.
    """
    scores = {}
    required_landmarks_group_A = ['left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist', 'neck']
    required_landmarks_group_B = ['neck', 'left_hip', 'left_knee', 'right_hip', 'right_knee']

    # Validação Mínima: checa se temos landmarks essenciais
    if not all(key in landmarks for key in required_landmarks_group_A) or not all(key in landmarks for key in required_landmarks_group_B):
        return None # Não podemos calcular se faltam pontos cruciais

    # --- Grupo A: Braços e Punhos ---
    # Média dos lados, como definido na regra de negócio
    upper_arm_angle = np.mean([
        rula_calculator.calculate_angle(landmarks['left_hip'], landmarks['left_shoulder'], landmarks['left_elbow']),
        rula_calculator.calculate_angle(landmarks['right_hip'], landmarks['right_shoulder'], landmarks['right_elbow'])
    ])
    scores['upper_arm'] = rula_calculator.classify_upper_arm(upper_arm_angle)

    lower_arm_angle = np.mean([
        rula_calculator.calculate_angle(landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']),
        rula_calculator.calculate_angle(landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist'])
    ])
    scores['lower_arm'] = rula_calculator.classify_lower_arm(lower_arm_angle)

    # Para o punho, a lógica pode ser mais complexa, mas usamos a base por enquanto
    wrist_angle = 15 # Assumindo neutro como placeholder, o cálculo real precisa de mais pontos
    scores['wrist'] = rula_calculator.classify_wrist(wrist_angle)

    # Usando nossa função de twist aprimorada
    scores['wrist_twist'] = rula_calculator.classify_wrist_twist_enhanced(landmarks)

    # --- Grupo B: Pescoço, Tronco e Pernas ---
    neck_angle = rula_calculator.calculate_angle(landmarks['nose'], landmarks['neck'], landmarks['left_hip']) # Aproximação
    scores['neck'] = rula_calculator.classify_neck(neck_angle)

    trunk_angle = rula_calculator.calculate_angle(landmarks['neck'], landmarks['left_hip'], landmarks['left_knee']) # Aproximação
    scores['trunk'] = rula_calculator.classify_trunk(trunk_angle)

    # Pernas (simplificado: se os dois tornozelos estão detectados, assumimos suporte)
    scores['legs'] = 1 if 'left_ankle' in landmarks and 'right_ankle' in landmarks else 2

    # --- Fatores Adicionais (fixos, conforme regra de negócio) ---
    scores['muscle_use'] = 1
    scores['force_load_a'] = 0
    scores['force_load_b'] = 0
    
    return scores

def process_video_rula(input_path: str, output_path: str, pose_model: str = 'openpose'):
    print("Iniciando o pipeline de processamento RULA...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 30)
        writer = imageio.get_writer(output_path, fps=fps)
    except Exception as e:
        print(f"Erro ao abrir os arquivos de vídeo: {e}")
        return

    frame_count = 0
    try:
        for frame_original in reader:
            frame_count += 1
            print(f"Processando frame {frame_count}...")

            frame_bgr = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)
            landmarks = pose_estimation.estimate_pose(frame_bgr, model_name=pose_model)
            annotated_frame = frame_bgr.copy()

            if landmarks:
                scores = calculate_scores_from_landmarks(landmarks)
                
                if scores:
                    final_score, risk_level = rula_calculator.rula_risk(scores)
                    
                    # --- CHAMANDO O REPORT_GENERATOR ---
                    # A responsabilidade de desenhar agora é delegada
                    annotated_frame = report_generator.draw_skeleton(annotated_frame, landmarks)
                    annotated_frame = report_generator.draw_rula_score(annotated_frame, final_score, risk_level)

            frame_to_write = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_to_write)

    finally:
        print("Finalizando e salvando o vídeo...")
        reader.close()
        writer.close()
        print(f"Processamento concluído. Vídeo salvo em: {output_path}")