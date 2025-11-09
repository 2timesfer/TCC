"""
Módulo Gerador de Relatório (Analisador Estatístico)

Responsável por:
1. Ler o arquivo .json de dados brutos (gerado pelo video_process.py).
2. Usar o Pandas para calcular estatísticas detalhadas sobre os dados
   (Média, Mediana, Desvio Padrão, Variância, Min, Max).
3. Formatar um relatório em texto (string) com essas estatísticas.
"""
import json
import pandas as pd
import logging
from typing import Dict, Any

def generate_report_from_raw_file(raw_json_path: str, user_context: Dict[str, Any], request_id: str) -> str:
    """
    Carrega um arquivo JSON de dados brutos (frame a frame) e calcula
    as estatísticas (Média, Mediana, DP, Variância, Min, Max)
    para os ângulos e scores, gerando um relatório em texto.
    
    Args:
        raw_json_path (str): O caminho para o arquivo _raw_data.json.
        user_context (dict): O dicionário de contexto do usuário.
        request_id (str): O ID da requisição para o relatório.

    Returns:
        str: Uma string formatada com o relatório estatístico.
    """
    try:
        # Carregar os dados brutos do JSON
        with open(raw_json_path, 'r') as f:
            data = json.load(f)
            
        if not data:
            return "Erro: O arquivo de dados brutos está vazio."

        # Converter para DataFrame do Pandas para análise fácil
        df = pd.DataFrame(data)

        # --- Extrair Contexto ---
        job = user_context.get("job_role", "N/A")
        hours = user_context.get("hours_per_day", "N/A")

        report_lines = [
            f"Relatório de Análise Ergonômica (RULA)",
            f"ID da Análise: {request_id}",
            "------------------------------------------------",
            "Contexto do Usuário:",
            f"* Cargo: {job}",
            f"* Horas por Dia: {hours}",
            "------------------------------------------------",
            f"Resumo da Análise (Baseado em {len(df)} frames)",
        ]

        # Colunas que queremos analisar
        cols_to_analyze = {
            'rula_score': 'Score RULA',
            'neck_angle': 'Ângulo do Pescoço (graus)',
            'trunk_angle': 'Ângulo do Tronco (graus)',
            'upper_arm_angle': 'Ângulo do Braço Superior (graus)',
            'lower_arm_angle': 'Ângulo do Braço Inferior (graus)'
        }

        for col_name, display_name in cols_to_analyze.items():
            if col_name not in df.columns:
                report_lines.append(f"\nAVISO: Coluna '{display_name}' não encontrada nos dados.")
                continue

            # Converter para numérico, tratando erros (como 'NULL')
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            # Remover NaNs (frames onde a detecção pode ter falhado)
            df_valid = df.dropna(subset=[col_name])

            if df_valid.empty:
                report_lines.append(f"\nEstatísticas para: {display_name}")
                report_lines.append("  - (Nenhum dado válido encontrado)")
                continue

            # Calcular as estatísticas que você pediu
            mean = df_valid[col_name].mean()
            median = df_valid[col_name].median()
            std_dev = df_valid[col_name].std()
            variance = df_valid[col_name].var()
            min_val = df_valid[col_name].min()
            max_val = df_valid[col_name].max()

            # Adicionar ao relatório
            report_lines.append(f"\nEstatísticas para: {display_name}")
            report_lines.append(f"  - Média: {mean:.2f}")
            report_lines.append(f"  - Mediana: {median:.2f}")
            report_lines.append(f"  - Desvio Padrão (DP): {std_dev:.2f}")
            report_lines.append(f"  - Variância: {variance:.2f}")
            report_lines.append(f"  - Mínimo: {min_val:.2f}")
            report_lines.append(f"  - Máximo: {max_val:.2f}")
        
        report_lines.append("------------------------------------------------")
            
        return "\n".join(report_lines)
    
    except FileNotFoundError:
        logging.error(f"Arquivo de dados brutos não encontrado: {raw_json_path}")
        return f"Erro: Arquivo de dados brutos não encontrado em {raw_json_path}"
    except Exception as e:
        logging.error(f"Erro ao gerar relatório estatístico: {e}")
        return f"Erro ao processar dados estatísticos: {e}"