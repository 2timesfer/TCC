from core.pipeline import process_video_rula

def main():
    """
    Ponto de entrada principal para a aplicação.
    Define os caminhos de entrada e saída e chama o pipeline.
    """
    print("Aplicação de Análise RULA iniciada.")
    
    # Defina aqui os caminhos para os seus vídeos de teste
    video_de_entrada = r'C:\Users\ferca\Downloads\TCC\data\S17_A05_T01 (1).mp4'
    video_de_saida = 'resultados/video_analisado.mp4'
    
    # Chama a função principal do pipeline
    # Note que você pode escolher o modelo aqui ('openpose' ou 'mediapipe')
    process_video_rula(
        input_path=video_de_entrada,
        output_path=video_de_saida,
        pose_model='openpose'
    )

if __name__ == "__main__":
    # Esta linha garante que a função main() só será executada
    # quando você rodar o script diretamente (python main.py)
    main()