"""
Módulo Gerador de Relatório (Analisador Estatístico)

Responsável por:
1. Ler o arquivo .json de dados brutos (gerado pelo video_process.py).
2. Usar o Pandas para calcular estatísticas detalhadas sobre os dados
   (Média, Mediana, Desvio Padrão, Variância, Min, Max).
3. Formatar um relatório em texto (string) com essas estatísticas. """

import json
import pandas as pd
import logging
import ollama
from typing import Dict, Any

"""
def generate_report_from_raw_file(raw_json_path: str, user_context: Dict[str, Any], request_id: str) -> str: 

    Carrega um arquivo JSON de dados brutos (frame a frame) e calcula
    as estatísticas (Média, Mediana, DP, Variância, Min, Max)
    para os ângulos e scores, gerando um relatório em texto.
    
    Args:
        raw_json_path (str): O caminho para o arquivo _raw_data.json.
        user_context (dict): O dicionário de contexto do usuário.
        request_id (str): O ID da requisição para o relatório.

    Returns:
        str: Uma string formatada com o relatório estatístico.
    
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
        return f"Erro ao processar dados estatísticos: {e}" """
"""
Módulo Gerador de Relatório via AI (Ollama + Pandas)

Responsável por:
1. Ler o arquivo .json de dados brutos.
2. Calcular estatísticas matemáticas precisas via Pandas.
3. Enviar estatísticas + contexto para o Ollama.
4. Retornar um JSON estruturado com dados e sugestões de melhoria.
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula estatísticas matemáticas puras usando Pandas.
    Isso garante precisão numérica antes de passar para a IA.
    """
    stats = {}
    
    # Mapeamento de colunas técnicas para nomes legíveis
    cols_to_analyze = {
        'rula_score': 'Score RULA',
        'neck_angle': 'Ângulo do Pescoço',
        'trunk_angle': 'Ângulo do Tronco',
        'upper_arm_angle': 'Ângulo do Braço Superior',
        'lower_arm_angle': 'Ângulo do Braço Inferior'
    }

    for col_name, display_name in cols_to_analyze.items():
        if col_name not in df.columns:
            continue

        # Limpeza de dados
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        df_valid = df.dropna(subset=[col_name])

        if df_valid.empty:
            stats[display_name] = "Sem dados válidos"
            continue

        # Cálculo estatístico
        stats[display_name] = {
            "media": round(float(df_valid[col_name].mean()), 2),
            "mediana": round(float(df_valid[col_name].median()), 2),
            "desvio_padrao": round(float(df_valid[col_name].std()), 2),
            "maximo": round(float(df_valid[col_name].max()), 2),
            "minimo": round(float(df_valid[col_name].min()), 2)
        }
    
    return stats

def generate_report_json(raw_json_path: str, user_context: Dict[str, Any], request_id: str, model_name: str = "llama3.2") -> Dict[str, Any]:
    """
    JOB 1: GERAÇÃO DO JSON
    Lê o arquivo bruto, calcula estatísticas e usa IA para gerar análise em JSON.
    """
    try:
        # 1. Carregar dados brutos
        with open(raw_json_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            return {"error": "Arquivo de dados vazio"}

        # 2. Calcular estatísticas "duras" (Hard Data)
        df = pd.DataFrame(data)
        statistics = calculate_statistics(df)

        # 3. Preparar Prompt para o Ollama
        job_role = user_context.get("job_role", "Não informado")
        hours = user_context.get("hours_per_day", "Não informado")

        system_prompt = (
            "Você é um Fisioterapeuta Especialista em Ergonomia e método RULA. "
            "Sua tarefa é analisar os dados estatísticos fornecidos e gerar um relatório em JSON. "
            "O tom deve ser profissional, mas amigável e encorajador. "
            "O IDIOMA DEVE SER PORTUGUÊS DO BRASIL."
        )

        user_prompt = f"""
        Contexto do Usuário:
        - Cargo: {job_role}
        - Horas de trabalho/dia: {hours}
        - Total de frames analisados: {len(df)}

        Dados Estatísticos (Ângulos e Scores):
        {json.dumps(statistics, indent=2)}

        Tarefa:
        Gere um JSON estrito (sem markdown) com a seguinte estrutura:
        {{
            "resumo_executivo": "Um parágrafo resumindo o nível de risco geral.",
            "analise_segmentada": {{
                "pescoco": "Comentário específico sobre o pescoço",
                "tronco": "Comentário específico sobre o tronco",
                "bracos": "Comentário específico sobre os braços"
            }},
            "sugestoes_acao": [
                "Sugestão prática 1 (exercício ou ajuste)",
                "Sugestão prática 2",
                "Sugestão prática 3"
            ]
        }}
        """

        # 4. Chamada ao Ollama
        logger.info(f"Enviando dados para o Ollama ({model_name})...")
        response = ollama.chat(model=model_name, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ])

        ai_content = response['message']['content']

        # 5. Limpeza e Parse do JSON
        clean_json_str = ai_content.strip().replace("```json", "").replace("```", "")
        
        try:
            ai_analysis = json.loads(clean_json_str)
        except json.JSONDecodeError:
            # Fallback caso a IA retorne texto misturado
            logger.error("Falha ao processar JSON da IA. Retornando texto bruto.")
            ai_analysis = {
                "resumo_executivo": "Erro ao estruturar análise da IA.",
                "raw_text": ai_content,
                "sugestoes_acao": []
            }

        # 6. Montagem do Objeto Final
        final_report_data = {
            "meta": {
                "request_id": request_id,
                "job_role": job_role,
                "hours": hours,
                "total_frames": len(df)
            },
            "statistics": statistics,
            "ai_analysis": ai_analysis
        }

        return final_report_data

    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {raw_json_path}")
        return {"error": f"Arquivo não encontrado: {raw_json_path}"}
    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        return {"error": str(e)}

def format_report_for_gui(report_data: Dict[str, Any]) -> str:
    """
    JOB 2: EXIBIÇÃO NA GUI
    Transforma o JSON complexo em um texto bonito e legível para o usuário final (string).
    """
    if "error" in report_data:
        return f"⚠️ Erro ao gerar relatório: {report_data['error']}"

    meta = report_data.get("meta", {})
    stats = report_data.get("statistics", {})
    ai = report_data.get("ai_analysis", {})
    
    # Tentar pegar os campos da IA (com fallbacks seguros)
    resumo = ai.get("resumo_executivo", "Análise não disponível.")
    segmentos = ai.get("analise_segmentada", {})
    sugestoes = ai.get("sugestoes_acao", [])

    # Construção do Texto Formatado
    lines = []
    lines.append("========================================================")
    lines.append(f"📄 RELATÓRIO DE ANÁLISE ERGONÔMICA (ID: {meta.get('request_id', 'N/A')})")
    lines.append("========================================================")
    lines.append(f"👤 Cargo: {meta.get('job_role')} | ⏱️ Carga Horária: {meta.get('hours')}")
    lines.append("")
    
    lines.append("🔍 RESUMO EXECUTIVO")
    lines.append(f"{resumo}")
    lines.append("")
    
    lines.append("📊 DETALHES TÉCNICOS (Médias)")
    # Exibe apenas alguns dados chave das estatísticas para não poluir
    if "Score RULA" in stats:
        rula = stats["Score RULA"]
        lines.append(f"   • RULA Médio: {rula.get('media')} (Máx: {rula.get('maximo')})")
    if "Ângulo do Pescoço" in stats:
        lines.append(f"   • Pescoço: {stats['Ângulo do Pescoço'].get('media')}°")
    if "Ângulo do Tronco" in stats:
        lines.append(f"   • Tronco: {stats['Ângulo do Tronco'].get('media')}°")
    lines.append("")

    lines.append("🧠 ANÁLISE BIOMECÂNICA (IA)")
    if isinstance(segmentos, dict):
        lines.append(f"   ➤ Pescoço: {segmentos.get('pescoco', 'N/A')}")
        lines.append(f"   ➤ Tronco:  {segmentos.get('tronco', 'N/A')}")
        lines.append(f"   ➤ Braços:  {segmentos.get('bracos', 'N/A')}")
    else:
        lines.append("   (Detalhes segmentados não disponíveis)")
    lines.append("")

    lines.append("💡 RECOMENDAÇÕES & EXERCÍCIOS")
    if sugestoes:
        for i, sug in enumerate(sugestoes, 1):
            lines.append(f"   {i}. {sug}")
    else:
        lines.append("   Nenhuma sugestão específica gerada.")
    
    lines.append("")
    lines.append("========================================================")
    lines.append("Nota: Este relatório é gerado por IA e visão computacional.")
    lines.append("Consulte um profissional de saúde para diagnósticos clínicos.")
    
    return "\n".join(lines)

# ==========================================
# Exemplo de Uso (Simulando a chamada na GUI)
# ==========================================
if __name__ == "__main__":
    # Caminho fictício
    raw_path = "_raw_data.json" 
    
    # 1. Contexto vindo da GUI
    user_ctx = {"job_role": "Desenvolvedor de Software", "hours_per_day": "8h"}
    
    # 2. Gerar o JSON (Backend)
    # Nota: Isso vai falhar se o arquivo _raw_data.json não existir.
    # json_result = generate_report_json(raw_path, user_ctx, "REQ-123")
    
    # 3. Se tivéssemos o resultado, formataríamos para a GUI assim:
    # gui_text = format_report_for_gui(json_result)
    # print(gui_text)