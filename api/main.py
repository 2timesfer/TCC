import gradio as gr
from video_process import process_video_and_get_rula

def video_analysis_interface(video_file, detector_choice): # <-- Parâmetro adicionado
    """
    Função que a interface do Gradio irá chamar.
    """
    if video_file is None:
        return "Por favor, envie um vídeo."
    
    output_video_path = "rula_processed_video.mp4"
    
    # Converte a escolha do Radio para o nome que o backend espera
    detector_name = detector_choice.lower()
    
    processed_video = process_video_and_get_rula(video_file, output_video_path, detector_name)
    
    return processed_video

if __name__ == "__main__":
    iface = gr.Interface(
        fn=video_analysis_interface,
        inputs=[
            gr.Video(label="Upload do Vídeo para Análise"),
            gr.Radio(
                ["MediaPipe", "OpenPose", "YOLO"], 
                label="Escolha o Detector de Pose", 
                value="MediaPipe" # Valor padrão
            )
        ],
        outputs=gr.Video(label="Resultado da Análise"),
        title="Análise Ergonômica de Risco (RULA)",
        description="Faça o upload de um vídeo e escolha o modelo de detecção de pose para calcular o risco ergonômico."
    )
    
    iface.launch()