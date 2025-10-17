import gradio as gr
from video_process import process_video_and_get_rula

def video_analysis_interface(video_file):
    """
    Função que a interface do Gradio irá chamar.
    """
    if video_file is None:
        return "Por favor, envie um vídeo."
    
    output_video_path = r"C:\Users\ferca\Downloads\TCC\resultados\rula_processed_video.mp4"
    
    # Chama a função principal de processamento (chamada simplificada)
    processed_video = process_video_and_get_rula(video_file, output_video_path)    
    return processed_video

if __name__ == "__main__":
    iface = gr.Interface(
        fn=video_analysis_interface,
        inputs=gr.Video(label="Upload do Vídeo para Análise Ergonômica"),
        outputs=gr.Video(label="Resultado da Análise"),
        title="Análise Ergonômica de Risco (RULA)",
        description="Faça o upload de um vídeo para processar e calcular o risco ergonômico."
    )
    
    # Para rodar localmente
    iface.launch()