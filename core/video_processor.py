import imageio.v2 as imageio
import numpy as np
import cv2
from typing import List

def read_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Lê um arquivo de vídeo e retorna uma lista de frames.

    Args:
        video_path (str): O caminho para o arquivo de vídeo.

    Returns:
        List[np.ndarray]: Uma lista onde cada item é um frame do vídeo no formato
                          de um array NumPy (em BGR, padrão do OpenCV).
    
    Raises:
        FileNotFoundError: Se o caminho do vídeo não for encontrado.
        Exception: Para outros erros de leitura de vídeo.
    """
    try:
        reader = imageio.get_reader(video_path)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in reader]
        print(f"{len(frames)} frames lidos de {video_path}")
        return frames
    except FileNotFoundError:
        print(f"ERRO: Arquivo de vídeo não encontrado em '{video_path}'")
        raise
    except Exception as e:
        print(f"ERRO: Não foi possível ler o vídeo. Detalhes: {e}")
        raise

def write_video_frames(output_path: str, frames: List[np.ndarray], fps: int = 30):
    """
    Escreve uma lista de frames para um novo arquivo de vídeo.

    Args:
        output_path (str): O caminho onde o novo vídeo será salvo.
        frames (List[np.ndarray]): A lista de frames a ser escrita.
        fps (int): A taxa de quadros por segundo para o vídeo de saída.
    """
    if not frames:
        print("AVISO: Nenhuma frame para escrever. O vídeo não será gerado.")
        return

    print(f"Escrevendo {len(frames)} frames para {output_path} com {fps} FPS...")
    try:
        # imageio espera frames em RGB, então convertemos de BGR (nosso padrão) para RGB
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
    finally:
        writer.close()
    print("Vídeo gerado com sucesso.")