"""
Estandarización de carga de entrenamiento en clavados.

ADVERTENCIA DE LIMITACIÓN CIENTÍFICA (C4):
    Este módulo implementa una propuesta de diseño original para cuantificar
    la carga en clavados competitivos. Los coeficientes k_alt, k_DD, k_tipo y
    k_angulo, así como la fórmula de Carga Integrada (CI), no han sido
    validados prospectivamente en una cohorte de clavadistas. Su uso en
    decisiones clínicas individuales requiere validación empírica previa.

Coeficiente de altura (k_alt) — default beta=1.0:
    Pandey et al. (Sci. Adv. 2022, DOI: 10.1126/sciadv.abo5888) midieron
    directamente las fuerzas de impacto en clavados humanos a distintas alturas
    y demostraron que el impulso hidrodinámico promediado escala linealmente
    con la altura de caída. Por tanto k_alt(h) = h / h_ref (beta=1.0).

    beta=0.5 disponible como límite inferior conservador para análisis de
    sensibilidad únicamente (modelo invalidado por Pandey 2022).

Coeficiente de angulo (k_angulo):
    Fundamento conceptual: Harrison SM et al., Appl. Math. Model.
    2016;40:1551-1568. DOI: 10.1016/j.apm.2015.07.021

Carga Integrada (CI):
    CI = L_norm x (2 - W_norm) es una propuesta de diseno original.
    La logica general de que la carga interna depende del estado de recuperacion
    esta documentada en Halson (2014, Sports Med,
    DOI: 10.1007/s40279-014-0253-z), pero la funcion especifica CI = L x (2-W)
    requiere validacion prospectiva antes de uso clinico.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Tabla precalculada de k_alt con beta=1.0 (Pandey 2022) para instalaciones FINA.
K_ALT_TABLE: dict[float, float] = {
    1.0: 1.000,
    3.0: 3.000,
    5.0: 5.000,
    7.5: 7.500,
    10.0: 10.000,
}

#: Grado de dificultad de referencia (clasificacion estandar FINA).
DD_REF: float = 2.0

#: Coeficientes por tipo de entrada al agua.
K_TIPO: dict[str, float] = {
    "HEAD":  1.00,   # cabeza primero — referencia
    "FEET":  0.85,   # pies primero — menor carga lumbar
    "TWIST": 1.20,   # con twists (>2 giros) — carga torsional adicional
    "PIKE":  1.10,   # con piruetas — tension isquiotibiales/lumbar
    "SYNC":  1.15,   # sincronizado — estres psicomotor adicional
}

#: Maximo de referencia para normalizacion (95th percentil sesion elite, beta=1.0).
#: 20 clavados x k_alt(10m)=10.0 x k_dd(4.4/2.0)=2.2 x k_tipo(TWIST)=1.20 = 528.
#: L_MAX_REFERENCIA=500 cubre el 95th percentil con margen de seguridad.
#: k_angulo NO se incluye para mantener compatibilidad con sesiones sin datos de angulo.
L_MAX_REFERENCIA: float = 500.0


# ---------------------------------------------------------------------------
# Funciones de coeficientes
# ---------------------------------------------------------------------------

def k_alt(altura: float, beta: float = 1.0) -> float:
    """
    Coeficiente de equivalencia de altura.

    Formula: k_alt(h) = (h / h_ref) ** beta   donde h_ref = 1.0 m.

    Default beta=1.0 — modelo de impulso lineal (Pandey et al., Sci. Adv. 2022,
    DOI: 10.1126/sciadv.abo5888).

    Modelos alternativos disponibles via parametro beta:
      - beta=0.75: modelo empirico intermedio; requiere calibracion con
                   plataforma de fuerza.
      - beta=0.5:  limite inferior conservador; fue el modelo original,
                   invalidado por Pandey 2022. Solo para analisis de sensibilidad.

    Args:
        altura: Altura de la plataforma o trampolin en metros (> 0).
        beta:   Exponente del modelo (default 1.0).

    Returns:
        Coeficiente adimensional.

    Raises:
        ValueError: si altura <= 0.
    """
    if altura <= 0:
        raise ValueError(f"altura debe ser > 0, recibido: {altura}")
    return (altura / 1.0) ** beta


def k_dd(dd: float) -> float:
    """
    Coeficiente de grado de dificultad FINA.

    k_DD(DD) = DD / DD_ref   donde DD_ref = 2.0.

    Args:
        dd: Grado de dificultad FINA en [1.2, 4.4].

    Returns:
        Coeficiente en [0.60, 2.20].

    Raises:
        ValueError: si dd esta fuera del rango FINA [1.2, 4.4].
    """
    if not (1.2 <= dd <= 4.4):
        raise ValueError(f"DD debe estar en [1.2, 4.4] (FINA). Recibido: {dd}")
    return dd / DD_REF


def k_tipo(tipo: str) -> float:
    """
    Coeficiente por tipo de entrada al agua.

    Args:
        tipo: Codigo del tipo. Uno de HEAD, FEET, TWIST, PIKE, SYNC.

    Returns:
        Factor multiplicativo.

    Raises:
        ValueError: si el tipo no esta en K_TIPO.
    """
    if tipo not in K_TIPO:
        raise ValueError(f"tipo '{tipo}' no reconocido. Validos: {list(K_TIPO)}")
    return K_TIPO[tipo]


def k_angulo(angulo_grados: float | None) -> float:
    """
    Coeficiente de desviacion angular en la entrada al agua (C5).

    Formula: k_angulo(theta) = 1.0 + (theta / 45.0) x 0.30   theta en [0, 45] grados.

    Donde theta es la desviacion en grados respecto a la vertical
    (0 grados = entrada perfectamente vertical).

    Fundamento conceptual: Harrison SM et al., Appl. Math. Model.
    2016;40:1551-1568. DOI: 10.1016/j.apm.2015.07.021

    Args:
        angulo_grados: Desviacion de la vertical en grados [0, 45], o None.
                       Si None, retorna 1.0 (retrocompatible).

    Returns:
        Factor multiplicativo en [1.0, 1.3].

    Raises:
        ValueError: si angulo_grados esta fuera de [0, 45].
    """
    if angulo_grados is None:
        return 1.0
    if not (0.0 <= angulo_grados <= 45.0):
        raise ValueError(
            f"angulo_grados debe estar en [0, 45], recibido: {angulo_grados}"
        )
    return 1.0 + (angulo_grados / 45.0) * 0.30


# ---------------------------------------------------------------------------
# Carga bruta de sesion
# ---------------------------------------------------------------------------

def carga_bruta_sesion(
    clavados: list[dict],
    beta: float = 1.0,
) -> float:
    """
    Calcula la carga bruta de una sesion de clavados.

    Contribucion por clavado:
        contribucion_i = k_alt(h_i, beta) x k_dd(DD_i) x k_tipo(tipo_i) x k_angulo(theta_i)

    L_sesion = suma de contribucion_i

    Cada dict en clavados debe tener:
        - "altura":         float — altura en metros (requerido)
        - "dd":             float — grado de dificultad FINA (requerido)
        - "tipo":           str   — HEAD/FEET/TWIST/PIKE/SYNC (requerido)
        - "angulo_grados":  float | None — desviacion vertical en grados (opcional)

    k_angulo NO se incluye en L_MAX_REFERENCIA para mantener compatibilidad
    con sesiones sin datos de angulo.

    Args:
        clavados: Lista de dicts describiendo cada clavado ejecutado.
        beta:     Exponente del modelo k_alt (default 1.0).

    Returns:
        Carga bruta L_sesion como float.

    Raises:
        ValueError: si falta una clave requerida o los valores estan fuera de rango.
    """
    total = 0.0
    for i, clavado in enumerate(clavados):
        try:
            h    = clavado["altura"]
            dd   = clavado["dd"]
            tipo = clavado["tipo"]
        except KeyError as e:
            raise ValueError(f"Clavado {i} falta la clave: {e}") from e
        if tipo not in K_TIPO:
            raise ValueError(f"Tipo de entrada '{tipo}' invalido. Opciones: {list(K_TIPO)}")
        ka   = k_alt(h, beta)
        kd   = k_dd(dd)
        kt   = K_TIPO[tipo]
        kang = k_angulo(clavado.get("angulo_grados", None))
        total += ka * kd * kt * kang
    return total


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------

def normalizar_carga(l_sesion: float) -> float:
    """
    Normaliza la carga bruta a escala [0, 100] usando L_MAX_REFERENCIA=500.

    L_norm = min(L_sesion / L_MAX_REFERENCIA x 100, 100)

    Args:
        l_sesion: Carga bruta de la sesion (float >= 0).

    Returns:
        Carga normalizada en [0.0, 100.0].
    """
    return min(l_sesion / L_MAX_REFERENCIA * 100.0, 100.0)


# ---------------------------------------------------------------------------
# Wellness
# ---------------------------------------------------------------------------

def calcular_wellness(
    sueno: int,
    fatiga: int,
    estres: int,
    dolor: int,
    humor: int,
) -> float:
    """
    Calcula el indice de wellness normalizado W_norm en [0, 1] basado en el
    Cuestionario de Hooper Modificado (5 items, escala Likert 1-7).

    Items descendentes (1=optimo): sueno, fatiga, estres, dolor.
        w_i = (7 - valor) / 6.0

    Item ascendente (7=optimo): humor.
        w_humor = (valor - 1) / 6.0

    W_norm = media aritmetica de los 5 componentes normalizados.
    W_norm = 1.0 optimo | W_norm = 0.0 pesimo.

    Args:
        sueno:  Calidad de sueno   (1=optimo, 7=pesimo).
        fatiga: Nivel de fatiga    (1=optimo, 7=pesimo).
        estres: Nivel de estres    (1=optimo, 7=pesimo).
        dolor:  Dolor muscular     (1=optimo, 7=pesimo).
        humor:  Estado de humor    (7=optimo, 1=pesimo).

    Returns:
        W_norm en [0.0, 1.0].

    Raises:
        ValueError: si algun item esta fuera del rango Hooper [1, 7].
    """
    items = {"sueno": sueno, "fatiga": fatiga, "estres": estres,
             "dolor": dolor, "humor": humor}
    for nombre, val in items.items():
        if not (1 <= val <= 7):
            raise ValueError(
                f"Item '{nombre}' fuera del rango Hooper [1, 7]. Recibido: {val}"
            )
    w_sueno  = (7 - sueno)  / 6.0
    w_fatiga = (7 - fatiga) / 6.0
    w_estres = (7 - estres) / 6.0
    w_dolor  = (7 - dolor)  / 6.0
    w_humor  = (humor - 1)  / 6.0
    return (w_sueno + w_fatiga + w_estres + w_dolor + w_humor) / 5.0


# ---------------------------------------------------------------------------
# Carga integrada
# ---------------------------------------------------------------------------

def carga_integrada(l_norm: float, w_norm: float) -> float:
    """
    Variable unificada de carga integrada (CI) — propuesta de diseno original.

    CI = L_norm x (2 - W_norm)

    Interpretacion del factor de amplificacion (2 - W_norm):
      - W_norm = 1.0 (optimo): factor = 1.0 — sin amplificacion
      - W_norm = 0.5 (regular): factor = 1.5 — carga amplificada 50%
      - W_norm = 0.0 (pesimo):  factor = 2.0 — carga percibida doble

    Rango: CI en [0, 200] cuando L_norm=100 y W_norm=0.

    NOTA METODOLOGICA (C3): Esta funcion es una propuesta de diseno original
    sin calibracion empirica directa en clavados. Requiere validacion
    prospectiva antes de uso clinico. Ver Halson (2014, Sports Med,
    DOI: 10.1007/s40279-014-0253-z) para el marco conceptual general.

    Args:
        l_norm: Carga normalizada en [0, 100].
        w_norm: Indice de wellness en [0, 1].

    Returns:
        CI en [0, 200].
    """
    return l_norm * (2.0 - w_norm)
