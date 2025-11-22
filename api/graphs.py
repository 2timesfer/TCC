import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- CONFIGURAÇÕES ---
PASTA_ENTRADA = 'resultados_do_lote'
PASTA_SAIDA = 'comparativo_resultados'

sns.set_theme(style="whitegrid")

def carregar_dados(caminho_arquivo, nome_detector):
    """
    Carrega o JSON, converte para DataFrame e TRATA VALORES NULOS ('NULL').
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        if not isinstance(dados, list):
            return None

        df = pd.DataFrame(dados)
        
        if 'rula_score' not in df.columns:
            return None

        # --- CORREÇÃO DO ERRO 'NULL' ---
        # Força a coluna a ser numérica. 
        # errors='coerce' transforma 'NULL', texto ou erros em NaN (vazio matemático)
        df['rula_score'] = pd.to_numeric(df['rula_score'], errors='coerce')
        
        # Opcional: Se você quiser preencher os buracos com 0 ou com a média, faria aqui.
        # Mas deixar como NaN é melhor pois o gráfico vai mostrar uma falha na linha,
        # indicando visualmente que o detector perdeu o alvo naquele momento.

        df['frame'] = df.index + 1
        df['detector'] = nome_detector
        return df

    except Exception as e:
        print(f"Erro ao ler {os.path.basename(caminho_arquivo)}: {e}")
        return None

def gerar_graficos(df_yolo, df_mp, video_nome, pasta_destino):
    # Sincroniza o tamanho dos vídeos (corta o excesso do maior)
    min_len = min(len(df_yolo), len(df_mp))
    d1 = df_yolo.iloc[:min_len].copy()
    d2 = df_mp.iloc[:min_len].copy()

    # 1. Evolução do Score
    plt.figure(figsize=(12, 6))
    plt.plot(d1['frame'], d1['rula_score'], label='YOLO', alpha=0.8)
    plt.plot(d2['frame'], d2['rula_score'], label='MediaPipe', alpha=0.8, linestyle='--')
    plt.title(f"Comparativo RULA Score - {video_nome}")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.savefig(os.path.join(pasta_destino, "grafico_evolucao.jpg"))
    plt.close()

    # 2. Diferença (Delta)
    plt.figure(figsize=(12, 4))
    plt.bar(d1['frame'], d1['rula_score'] - d2['rula_score'], color='gray', alpha=0.5)
    plt.title(f"Diferença (YOLO - MediaPipe) - {video_nome}")
    plt.savefig(os.path.join(pasta_destino, "grafico_diferenca.jpg"))
    plt.close()

    return {
        "Video": video_nome,
        "Media_YOLO": d1['rula_score'].mean(),
        "Media_MP": d2['rula_score'].mean(),
        "Correlacao": d1['rula_score'].corr(d2['rula_score'])
    }

def main():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    # Dicionário para agrupar pares: grupos[id_video] = {'yolo': caminho, 'mediapipe': caminho}
    grupos = {}
    
    # Busca todos os JSONs
    arquivos = glob.glob(os.path.join(PASTA_ENTRADA, "*.json"))
    print(f"Total de arquivos encontrados na pasta: {len(arquivos)}")

    for arquivo in arquivos:
        nome = os.path.basename(arquivo).lower()

        # --- FILTROS ---
        # 1. Ignora se for relatório
        if 'relatorio' in nome:
            continue
        
        # 2. Tenta identificar o tipo (YOLO ou MediaPipe)
        tipo = None
        if 'yolo' in nome:
            tipo = 'yolo'
            # Extrai o ID: pega tudo antes de '_yolo'
            # Ex: "20251119_155352_yolo_dados_brutos.json" -> ID: "20251119_155352"
            vid_id = nome.split('_yolo')[0]
        elif 'mediapipe' in nome:
            tipo = 'mediapipe'
            vid_id = nome.split('_mediapipe')[0]
        
        # Se identificou um tipo válido, adiciona ao grupo
        if tipo:
            if vid_id not in grupos:
                grupos[vid_id] = {}
            grupos[vid_id][tipo] = arquivo

    # Processa os pares encontrados
    resumo = []
    print(f"Pares de vídeos identificados: {len(grupos)}")

    for vid_id, paths in grupos.items():
        # Só processa se tiver os dois arquivos (YOLO e MediaPipe)
        if 'yolo' in paths and 'mediapipe' in paths:
            print(f"Processando: {vid_id}...")
            
            df_yolo = carregar_dados(paths['yolo'], "YOLO")
            df_mp = carregar_dados(paths['mediapipe'], "MediaPipe")

            if df_yolo is not None and df_mp is not None:
                # Cria pasta específica para esse vídeo
                subpasta = os.path.join(PASTA_SAIDA, f"analise_{vid_id}")
                if not os.path.exists(subpasta):
                    os.makedirs(subpasta)
                
                # Gera gráficos
                metricas = gerar_graficos(df_yolo, df_mp, vid_id, subpasta)
                resumo.append(metricas)
        else:
            # Opcional: Avisar se falta algum par
            pass

    # Salva tabela resumo final
    if resumo:
        pd.DataFrame(resumo).to_csv(os.path.join(PASTA_SAIDA, "resumo_geral.csv"), index=False)
        print("\nProcessamento concluído com sucesso!")
    else:
        print("\nNenhum par de dados brutos (YOLO + MediaPipe) foi encontrado para processar.")

if __name__ == "__main__":
    main()