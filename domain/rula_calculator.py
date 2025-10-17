import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# --- Carregamento das Tabelas RULA ---
# Em uma arquitetura final, este caminho pode vir de um arquivo de configuração.
try:
    TABLE_A = pd.read_csv('data/TableA.csv')
    TABLE_B = pd.read_csv('data/TableB.csv')
    TABLE_C = pd.read_csv('data/TableC.csv')
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

def classify_wrist_twist_enhanced(landmarks: Dict[str, Tuple[int, int]]) -> int:
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

def rula_risk(scores: Dict[str, Any], point_score: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """
    Calcula o score e o nível de risco RULA a partir de um dicionário `scores`.
    Retorna (final_score, risk_level).

    scores: dicionário esperado (algumas chaves possíveis):
        - wrist
        - trunk
        - upper_Shoulder  OR upper_arm
        - lower_Limb      OR lower_arm
        - neck
        - wrist_twist
        - legs
        - muscle_use (opcional, default 0)
        - force_load_a (opcional, default 0)
        - force_load_b (opcional, default 0)
        - upper_body_muscle (opcional, default 0)

    point_score: dicionário opcional que será preenchido (se fornecido).
    """
    global TABLE_A, TABLE_B, TABLE_C

    # inicializa point_score se não foi passado
    if point_score is None:
        point_score = {}

    # Valida presença das tabelas
    if TABLE_A is None or TABLE_B is None or TABLE_C is None:
        return 'NULL', 'Tabelas não carregadas'

    # --- Normaliza nomes de keys (aceita variações) ---
    wrist = scores.get('wrist')
    trunk = scores.get('trunk')
    upper_Shoulder = scores.get('upper_Shoulder', scores.get('upper_arm'))
    lower_Limb = scores.get('lower_Limb', scores.get('lower_arm'))
    neck = scores.get('neck')
    wrist_twist = scores.get('wrist_twist')
    legs = scores.get('legs')
    muscle_use = scores.get('muscle_use', 0)
    force_load_a = scores.get('force_load_a', 0)
    force_load_b = scores.get('force_load_b', 0)
    upper_body_muscle = scores.get('upper_body_muscle', 0)

    # Validação mínima
    if not all(v is not None and v != 0 for v in (wrist, trunk, upper_Shoulder, lower_Limb, neck, wrist_twist)):
        return 'NULL', 'Dados insuficientes'

    # --- Table A lookup ---
    colA = f"{wrist}WT{wrist_twist}"
    if colA not in TABLE_A.columns:
        return 'NULL', f"Coluna '{colA}' ausente em TABLE_A"

    subsetA = TABLE_A.loc[
        (TABLE_A.UpperArm == upper_Shoulder) &
        (TABLE_A.LowerArm == lower_Limb),
        colA
    ]
    if getattr(subsetA, "empty", True):
        return 'NULL', f"Combinação inválida na TABLE_A para UpperArm={upper_Shoulder}, LowerArm={lower_Limb}"

    try:
        valA_base = float(subsetA.iloc[0]) # type: ignore
    except Exception:
        return 'NULL', "Valor inválido em TABLE_A"

    point_score['posture_score_a'] = valA_base
    valA = valA_base + muscle_use + force_load_a
    point_score['wrist_and_arm_score'] = valA

    # --- Table B lookup ---
    colB = f"{trunk}{legs}"
    if colB not in TABLE_B.columns:
        return 'NULL', f"Coluna '{colB}' ausente em TABLE_B"

    subsetB = TABLE_B.loc[TABLE_B.Neck == neck, colB]
    if getattr(subsetB, "empty", True):
        return 'NULL', f"Combinação inválida na TABLE_B para Neck={neck}"

    try:
        valB_base = float(subsetB.iloc[0]) # type: ignore
    except Exception:
        return 'NULL', "Valor inválido em TABLE_B"

    point_score['posture_score_b'] = valB_base
    valB = valB_base + force_load_b + upper_body_muscle
    point_score['neck_trunk_leg_score'] = valB

    # --- Table C lookup ---
    a_idx = min(int(valA), 8)
    b_idx = min(int(valB), 7)
    colC = str(b_idx)
    subsetC = TABLE_C.loc[TABLE_C.Score == a_idx, colC] if colC in TABLE_C.columns else None

    if subsetC is None or getattr(subsetC, "empty", True):
        return 'NULL', f"Combinação inválida na TABLE_C (Score={a_idx}, coluna={colC})"

    try:
        valC = int(subsetC.iloc[0])
    except Exception:
        return 'NULL', "Valor inválido em TABLE_C"

    # --- Classificação ---
    if valC <= 2:
        risk = 'Negligible'
    elif valC <= 4:
        risk = 'Low risk'
    elif valC <= 6:
        risk = 'Medium risk'
    else:
        risk = 'Very high risk'

    return valC, risk