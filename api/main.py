"""
Módulo Principal (Interface Gradio)

Responsável por:
1. Criar a interface web com Gradio.
2. Receber o upload do vídeo e as configurações do usuário.
3. Orquestrar o fluxo de trabalho:
    a. Definir nomes de arquivos de saída (vídeo, json bruto).
    b. Chamar 'video_process.py' para coletar os dados.
    c. Chamar 'report_generator.py' para analisar os dados brutos.
4. Exibir os resultados (vídeo processado, link para baixar o JSON
   bruto, e o relatório estatístico em texto).
"""
import gradio as gr
from video_process import process_video_and_collect_raw_data
from reports import generate_report_from_raw_file 
import datetime
import json
import os

# Contexto do usuário (será passado para o gerador de relatório)
DEFAULT_USER_CONTEXT = {
    "job_role": "analista_marketing",
    "hours_per_day": 6,
    "age": 34,
    "sex": "Female",
    "self_reported_pain_area": "nenhuma",
    "self_reported_pain_level_0_10": 0
}

def video_analysis_interface(video_file, detector_choice):
    """
    Função que a interface do Gradio irá chamar.
    """
    if video_file is None:
        return None, None, "Por favor, envie um vídeo." 

    # --- 1. Configurar nomes de arquivos e IDs ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = "video"
    if hasattr(video_file, 'name'): 
        base_filename = os.path.basename(video_file.name).split('.')[0]
    
    request_id = f"{base_filename}_{timestamp}"
    output_video_path = f"{request_id}_processed.mp4"
    # Define o nome do arquivo de dados brutos que será criado
    output_raw_data_path = f"{request_id}_raw_data.json" 
    
    detector_name = detector_choice.lower()
    
    print(f"Iniciando análise com ID: {request_id}")

    # --- 2. Chamar o Coletor de Dados ---
    # Ele processa o vídeo E salva o JSON bruto
    processed_video, raw_data_path = process_video_and_collect_raw_data(
        video_path=video_file,
        output_path=output_video_path,
        detector_name=detector_name,
        raw_data_json_path=output_raw_data_path # Passa o caminho de saída
    )
    
    if raw_data_path:
        # --- 3. Chamar o Analisador de Dados ---
        # Ele lê o JSON bruto e gera o relatório em texto
        text_report = generate_report_from_raw_file(
            raw_json_path=raw_data_path,
            user_context=DEFAULT_USER_CONTEXT,
            request_id=request_id
        )
        
        print(f"Análise concluída. Vídeo: {processed_video}, Dados Brutos: {raw_data_path}")
        
        # --- 4. Retornar os 3 resultados para o Gradio ---
        return processed_video, raw_data_path, text_report
    else:
        print("A análise falhou.")
        return None, None, "A análise do vídeo falhou."

if __name__ == "__main__":
    iface = gr.Interface(
        fn=video_analysis_interface,
        inputs=[
            gr.Video(label="Upload do Vídeo para Análise"),
            gr.Radio(
                ["MediaPipe", "OpenPose", "YOLO"], 
                label="Escolha o Detector de Pose", 
                value="YOLO"
            )
        ],
        outputs=[
            gr.Video(label="Resultado da Análise"),
            gr.File(label="Baixar Dados Brutos Frame-a-Frame (.json)"), 
            gr.Textbox(label="Relatório Estatístico", lines=20) 
        ],
        title="Análise Ergonômica de Risco (RULA) - Coletor e Analisador",
        description="Faça o upload de um vídeo. O sistema irá processá-lo, gerar um vídeo de análise, um arquivo .json com os dados brutos de cada frame e um relatório estatístico."
    )
    
    iface.launch()