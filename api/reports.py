"""
Módulo de Geração de Relatórios

Este arquivo contém funções responsáveis por traduzir os dados
estatísticos brutos (JSON) em relatórios legíveis para humanos.

Pode ser expandido para gerar PDFs, HTML ou prompts para LLMs.
"""

import logging
from typing import Dict, Any

def generate_text_summary(summary_data: Dict[str, Any]) -> str:
    """
    Gera um resumo em texto simples a partir do JSON de estatísticas.
    
    Este texto pode ser usado para um PDF ou como prompt para um LLM.
    
    Args:
        summary_data (dict): O dicionário JSON completo gerado por 
                             'calculate_summary_statistics'.

    Returns:
        str: Uma string formatada com o relatório condensado.
    """
    try:
        # Extrair seções para facilitar a leitura
        metrics = summary_data.get("video_metrics", {})
        stats = summary_data.get("video_statics", {})
        context = summary_data.get("user_context", {})

        # Extrair valores-chave
        request_id = summary_data.get("request_id", "N/A")
        mean_score = stats.get("mean_rula_score", 0)
        assessment = stats.get("rula_score_assessment", "N/A")
        
        duration = metrics.get("total_video_duration_seconds", 0)
        bad_posture_time = metrics.get("time_in_bad_posture_seconds", 0)
        
        avg_head = metrics.get("avg_head_tilt_degrees", 0)
        max_head = metrics.get("max_head_tilt_degrees", 0)
        avg_back = metrics.get("avg_back_curvature_degrees", 0)
        max_back = metrics.get("max_back_curvature_degrees", 0)
        
        job = context.get("job_role", "N/A")
        hours = context.get("hours_per_day", "N/A")

        # Calcular percentual de tempo em má postura
        bad_posture_pct = 0
        if duration > 0:
            bad_posture_pct = (bad_posture_time / duration) * 100

        # Montar o texto
        report = f"""
Relatório de Análise Ergonômica (RULA)
ID da Análise: {request_id}
------------------------------------------------
Contexto do Usuário:
* Cargo: {job}
* Horas por Dia: {hours}
------------------------------------------------
Resumo do Risco:
* Score RULA Médio: {mean_score:.2f}
* Avaliação de Risco: {assessment}
------------------------------------------------
Métricas Chave do Vídeo:
* Duração Total: {duration:.1f} segundos
* Tempo em Postura de Risco: {bad_posture_time:.1f} segundos ({bad_posture_pct:.1f}%)
* Inclinação Média da Cabeça: {avg_head:.1f}° (Máx: {max_head:.1f}°)
* Curvatura Média das Costas: {avg_back:.1f}° (Máx: {max_back:.1f}°)
------------------------------------------------
"""
        # Remove espaços em branco extras das bordas
        return "\n".join([line.strip() for line in report.strip().split('\n')])
    
    except Exception as e:
        logging.error(f"Erro ao gerar resumo em texto: {e}")
        return f"Erro ao processar dados do resumo: {e}"