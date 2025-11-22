"""
Módulo de Processamento em Lote (Batch) - ATUALIZADO

Responsável por:
1. Iterar sobre vídeos em uma pasta.
2. Executar Visão Computacional (extração de dados).
3. Executar IA (Análise RULA + Sugestões via Ollama).
4. Salvar os resultados em 3 formatos:
   - Vídeo processado (.mp4)
   - Relatório Técnico Estruturado (.json)
   - Relatório Legível para Leitura (.txt)
"""
import os
import glob
import json
import logging

# Configuração de logs para o batch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importa as funções atualizadas
try:
    from video_process import process_video_and_collect_raw_data
    from reports import generate_report_json, format_report_for_gui
except ImportError:
    logging.error("Erro: 'video_process.py' ou 'reports.py' não encontrados no diretório.")
    exit(1)

# --- Configurações do Processamento ---

INPUT_FOLDER = "videos_para_processar"
OUTPUT_FOLDER = "resultados_do_lote"
DETECTOR_CHOICE = "yolo"  # Ou "mediapipe", "openpose"

# Contexto Padrão (No modo batch, assume-se o mesmo contexto ou edite aqui)
DEFAULT_USER_CONTEXT = {
    "job_role": "Operador de Produção",
    "hours_per_day": "8 horas",
    "age": 30,
    "sex": "Masculino",
    "self_reported_pain_area": "Ombro Direito",
    "self_reported_pain_level_0_10": 4
}

def run_batch_processing():
    """
    Executa o pipeline completo para todos os vídeos da pasta de entrada.
    """
    print("=== INICIANDO PROCESSAMENTO EM LOTE (COM IA) ===")
    print(f"📂 Entrada: {os.path.abspath(INPUT_FOLDER)}")
    print(f"📂 Saída:   {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"🤖 Modelo:  {DETECTOR_CHOICE.upper()} + Llama3")
    
    # Criar pastas se não existirem
    if not os.path.isdir(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Pasta de entrada criada. Coloque vídeos em '{INPUT_FOLDER}' e rode novamente.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Buscar vídeos
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
        
    if not video_files:
        print(f"❌ Nenhum vídeo encontrado em '{INPUT_FOLDER}'.")
        return
        
    print(f"📹 Total de vídeos: {len(video_files)}\n")

    # --- Loop Principal ---
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        print(f"▶️ [{i+1}/{len(video_files)}] Processando: {filename}")
        
        try:
            # 1. Definir Caminhos e IDs
            base_name = filename.split('.')[0]
            request_id = f"BATCH_{base_name}_{DETECTOR_CHOICE}"
            
            path_video_out = os.path.join(OUTPUT_FOLDER, f"{base_name}_{DETECTOR_CHOICE}_analisado.mp4")
            path_raw_json = os.path.join(OUTPUT_FOLDER, f"{base_name}_{DETECTOR_CHOICE}_dados_brutos.json")
            path_full_report_json = os.path.join(OUTPUT_FOLDER, f"{base_name}_{DETECTOR_CHOICE}_relatorio_ia.json")
            path_readable_txt = os.path.join(OUTPUT_FOLDER, f"{base_name}_{DETECTOR_CHOICE}_relatorio_final.txt")
            
            # 2. Etapa de Visão Computacional (Extração de Dados)
            print("   ↳ 👁️  Extraindo dados biomecânicos...")
            processed_vid, raw_data_path = process_video_and_collect_raw_data(
                video_path=video_path,
                output_path=path_video_out,
                detector_name=DETECTOR_CHOICE,
                raw_data_json_path=path_raw_json
            )
            
            if raw_data_path:
                # 3. Etapa de Inteligência Artificial (Geração do Relatório)
                print("   ↳ 🧠  Enviando estatísticas para Llama3...")
                report_json_data = generate_report_json(
                    raw_json_path=raw_data_path,
                    user_context=DEFAULT_USER_CONTEXT,
                    request_id=request_id
                )
                
                # Salva o JSON Completo da IA (Backup estruturado)
                with open(path_full_report_json, 'w', encoding='utf-8') as f:
                    json.dump(report_json_data, f, indent=2, ensure_ascii=False)

                # 4. Etapa de Formatação (Texto Legível)
                print("   ↳ 📄  Formatando relatório em texto...")
                readable_text = format_report_for_gui(report_json_data)
                
                # Salva o arquivo .txt bonito
                with open(path_readable_txt, "w", encoding="utf-8") as f:
                    f.write(readable_text)
                
                print(f"   ✅ Sucesso! Relatório salvo em: {path_readable_txt}")
            
            else:
                print(f" Falha na extração de dados do vídeo.")

        except Exception as e:
            logging.error(f"Erro ao processar {filename}: {e}")
            print(f"   ❌ Erro crítico no vídeo {filename}. Pulando...")

    print("\n=== Processamento em Lote Finalizado ===")

if __name__ == "__main__":
    run_batch_processing()