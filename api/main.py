import gradio as gr
from video_process import process_video_and_get_rula # MUDANÇA 1: Importar a nova função
import datetime
from reports import generate_text_summary
import json
import os

# MUDANÇA 2: Definir o contexto do usuário que será usado no teste
# Como a interface do Gradio não pede esses dados, vamos usar um valor padrão
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
    Atualizada para a nova função de processamento.
    """
    if video_file is None:
        # MUDANÇA 3: Retornar 2 Nones para os 2 outputs (Vídeo e Arquivo)
        return None, None 

    # --- Configurar nomes de arquivos e IDs ---
    # Usamos um timestamp para garantir que os arquivos sejam únicos
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Pega o nome do arquivo original para usar no ID
    base_filename = "video"
    if hasattr(video_file, 'name'): # Se for um arquivo temporário do Gradio
        base_filename = os.path.basename(video_file.name).split('.')[0]
    
    request_id = f"{base_filename}_{timestamp}"
    
    # Define os caminhos de saída
    output_video_path = f"{request_id}_processed.mp4"
    output_json_path = f"{request_id}_summary.json"
    
    detector_name = detector_choice.lower()
    
    print(f"Iniciando análise com ID: {request_id}")

    # MUDANÇA 4: Chamar a nova função 'process_video_and_generate_summary'
    # Ela precisa de 5 argumentos e retorna 2 valores
    processed_video, final_summary = process_video_and_get_rula(
        video_path=video_file,               # O vídeo do upload
        output_path=output_video_path,       # O caminho do vídeo de saída
        detector_name=detector_name,         # O detector escolhido
        user_context_dict=DEFAULT_USER_CONTEXT, # O contexto que definimos acima
        request_id=request_id                # O ID único que geramos
    )
    
    # MUDANÇA 5: Salvar o JSON retornado em um arquivo
    if final_summary:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=4, ensure_ascii=False)
        
        text_report = generate_text_summary(final_summary)

        print(f"Análise concluída. Vídeo: {processed_video}, Resumo: {output_json_path}")
        # Retorna o caminho do vídeo e o caminho do JSON
        return processed_video, output_json_path, text_report
    else:
        print("A análise falhou.")
        return None, None

if __name__ == "__main__":
    iface = gr.Interface(
        fn=video_analysis_interface,
        inputs=[
            gr.Video(label="Upload do Vídeo para Análise"),
            gr.Radio(
                ["MediaPipe", "OpenPose", "YOLO"], 
                label="Escolha o Detector de Pose", 
                value="YOLO" # Valor padrão (YOLO tende a ser mais robusto)
            )
        ],
        # MUDANÇA 6: Atualizar os 'outputs' para aceitar 2 resultados
        outputs=[
            gr.Video(label="Resultado da Análise"),
            gr.File(label="Baixar Resumo Estatístico (.json)"),
            gr.Textbox(label="Resumo Condensado", lines=15)
        ],
        title="Análise Ergonômica de Risco (RULA)",
        description="Faça o upload de um vídeo. O sistema irá processá-lo e gerar um vídeo de análise e um arquivo .json com as estatísticas."
    )
    
    iface.launch()