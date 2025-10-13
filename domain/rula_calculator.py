import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# --- Carregamento das Tabelas RULA ---
# Em uma arquitetura final, este caminho pode vir de um arquivo de configuração.
try:
    TABLE_A = pd.read_csv('/content/drive/MyDrive/TCC - Debs e Fer/TCC - Dados/rula_score_table_a.csv')
    TABLE_B = pd.read_csv('/content/drive/MyDrive/TCC - Debs e Fer/TCC - Dados/rula_score_table_b.csv')
    TABLE_C = pd.read_csv('/content/drive/MyDrive/TCC - Debs e Fer/TCC - Dados/rula_score_table_c.csv')
except FileNotFoundError:
    print("AVISO: A funcionalidade de cálculo de risco será limitada.")
    TABLE_A, TABLE_B, TABLE_C = None, None, None

# --- Funções de Cálculo de Ângulo ---

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (em graus)."""
    a = np.array(a)  # Primeiro ponto
    b = np.array(b)  # Ponto do meio (vértice)
    c = np.array(c)  # Terceiro ponto
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Funções de Classificação de Pontuação (Lógica RULA) ---

def classify_upper_arm(angle: float) -> int:
    """Classifica a pontuação do braço com base no ângulo."""
    if angle < 20:
        return 1
    if angle < 50:
        return 2
    if angle < 90:
        return 3
    return 4

def classify_lower_arm(angle: float) -> int:
    """Classifica a pontuação do antebraço com base no ângulo."""
    return 1 if 80<=angle<=100 else 2 if angle>100 else 1


def classify_wrist(angle: float) -> int:
    """Classifica a pontuação do punho com base no ângulo."""
    if angle == 15: # Posição neutra ideal
      return 1
    elif 0 <= angle < 15:
        return 2
    else: # angle > 15
        return 3

def classify_wrist_twist(angle):
    # < 15° → 1 ponto, senão → 2 pontos
    return 1 if angle < 15 else 2
    
def calculate_vector(p1, p2):
    """Calcula o vetor entre dois pontos."""
    return np.array(p2) - np.array(p1)

def classify_wrist_twist_enhanced(landmarks: Dict[str, Tuple[float, float]]) -> int:
    """
    Classifica a pontuação da torção do punho inferindo a orientação
    a partir da posição relativa do cotovelo e do punho.

    Esta é uma heurística para dados 2D. Assume-se que um desvio
    significativo na componente horizontal do vetor cotovelo-punho
    indica uma torção.

    Args:
        landmarks (Dict): Um dicionário contendo as coordenadas
                          dos pontos-chave, como 'left_elbow', 'left_wrist', etc.

    Returns:
        int: Pontuação de torção do punho (1 ou 2).
    """
    # Usaremos uma média dos dois lados, como definido na regra de negócio
    sides = ['left', 'right']
    twist_scores = []

    for side in sides:
        elbow = landmarks.get(f'{side}_elbow')
        wrist = landmarks.get(f'{side}_wrist')

        # Se não tivermos os landmarks para um dos lados, pulamos
        if elbow is None or wrist is None:
            continue

        # Calcula o vetor do antebraço
        forearm_vector = calculate_vector(elbow, wrist)

        # Normaliza o vetor para focar na direção
        norm = np.linalg.norm(forearm_vector)
        if norm == 0:
            twist_scores.append(1) # Posição neutra se não houver movimento
            continue
        
        forearm_unit_vector = forearm_vector / norm
        
        # Heurística: Verificamos o quanto o antebraço está "deitado" na horizontal.
        # Um valor alto em 'x' e baixo em 'y' sugere que a mão está em pronação
        # ou supinação (torcida). Usamos o valor absoluto do componente x.
        # Este limiar (threshold) pode ser ajustado. 0.7 representa um ângulo
        # de aproximadamente 45 graus com a vertical.
        horizontal_component = abs(forearm_unit_vector[0])
        
        if horizontal_component > 0.7:
            # Sugere que o antebraço está significativamente na horizontal,
            # o que é um bom indicador de torção em muitas tarefas.
            twist_scores.append(2)
        else:
            # Posição mais próxima da neutra (vertical).
            twist_scores.append(1)

    # Se não conseguimos calcular para nenhum lado, retornamos a pontuação neutra
    if not twist_scores:
        return 1

    # Retorna a média das pontuações (arredondada para o inteiro mais próximo)
    # como definido na regra de negócio para consolidação.
    final_score = round(np.mean(twist_scores))
    return int(final_score)

def classify_neck(angle: float) -> int:
    """Classifica a pontuação do pescoço com base no ângulo."""
    if 0 <= angle < 10:
        return 1
    elif 10 <= angle <= 20:
        return 2
    elif angle > 20:
        return 3
    else: # Em extensão
        return 4

def classify_trunk(angle: float) -> int:
    """Classifica a pontuação do tronco com base no ângulo."""
    if angle == 0: # Ereto
        return 1
    elif 0 < angle <= 20:
        return 2
    elif 20 < angle <= 60:
        return 3
    else: # angle > 60
        return 4

def classify_legs(is_supported: bool) -> int:
    """Classifica a pontuação das pernas."""
    return 1 if is_supported else 2

# --- Função Principal de Cálculo do Risco RULA ---

def rula_risk(scores: Dict[str, Any]) -> Tuple[int, str]:
    """
    Calcula a pontuação RULA final e o nível de risco a partir das pontuações individuais.
    
    Args:
        scores (Dict): Um dicionário contendo as pontuações individuais
                       (ex: 'upper_arm', 'neck', etc.).
                       
    Returns:
        Tuple[int, str]: Uma tupla contendo a pontuação final e a descrição do risco.
    """
    if TABLE_A is None or TABLE_B is None or TABLE_C is None:
        return -1, "Tabelas RULA não carregadas."

    # Tabela A: Pescoço, Tronco e Pernas
    wrist_arm_score = TABLE_A.loc[
        (TABLE_A['Upper Arm'] == scores['upper_arm']) &
        (TABLE_A['Lower Arm'] == scores['lower_arm']) &
        (TABLE_A['Wrist'] == scores['wrist']) &
        (TABLE_A['Wrist Twist'] == scores['wrist_twist']),
        'Score'
    ].iloc[0]
    
    wrist_arm_score += scores['muscle_use'] + scores['force_load_a']

    # Tabela B: Braço e Punho
    neck_trunk_leg_score = TABLE_B.loc[
        (TABLE_B['Neck'] == scores['neck']) &
        (TABLE_B['Trunk'] == scores['trunk']) &
        (TABLE_B['Legs'] == scores['legs']),
        'Score'
    ].iloc[0]

    neck_trunk_leg_score += scores['muscle_use'] + scores['force_load_b']
    
    # Tabela C: Pontuação Final
    final_score = TABLE_C.loc[
        (TABLE_C['Wrist_Arm_Score'] == wrist_arm_score) &
        (TABLE_C['Neck_Trunk_Leg_Score'] == neck_trunk_leg_score),
        'Score'
    ].iloc[0]
    
    # Classificação do Risco
    if final_score <= 2:
        risk = "Insignificant risk, no action required"
    elif 3 <= final_score <= 4:
        risk = "Low risk, change may be needed"
    elif 5 <= final_score <= 6:
        risk = "Medium risk, further investigation, change soon"
    else: # final_score >= 7
        risk = "Very high risk, investigate and implement change"
        
    return int(final_score), risk