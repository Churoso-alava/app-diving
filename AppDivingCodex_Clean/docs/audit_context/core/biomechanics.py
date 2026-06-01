"""
Estandarizacion de carga de entrenamiento en clavados.
"""

from __future__ import annotations

K_ALT_TABLE: dict[float, float] = {
    1.0: 1.000,
    3.0: 3.000,
    5.0: 5.000,
    7.5: 7.500,
    10.0: 10.000,
}

DD_REF: float = 2.0

K_TIPO: dict[str, float] = {
    "HEAD": 1.00,
    "FEET": 0.85,
    "TWIST": 1.20,
    "PIKE": 1.10,
    "SYNC": 1.15,
}

L_MAX_REFERENCIA: float = 500.0


def k_alt(altura: float, beta: float = 1.0) -> float:
    if altura <= 0:
        raise ValueError(f"altura debe ser > 0, recibido: {altura}")
    return (altura / 1.0) ** beta


def k_dd(dd: float) -> float:
    if not (1.2 <= dd <= 4.4):
        raise ValueError(f"DD debe estar en [1.2, 4.4] (FINA). Recibido: {dd}")
    return dd / DD_REF


def k_tipo(tipo: str) -> float:
    if tipo not in K_TIPO:
        raise ValueError(f"tipo '{tipo}' no reconocido. Validos: {list(K_TIPO)}")
    return K_TIPO[tipo]


def carga_bruta_sesion(
    clavados_planificados: list[dict],
    beta: float = 1.0,
) -> float:
    total = 0.0
    for i, clavado in enumerate(clavados_planificados):
        try:
            h = clavado["altura"]
            dd = clavado["dd"]
            tipo = clavado["tipo"]
        except KeyError as e:
            raise ValueError(f"Clavado {i} falta la clave: {e}") from e
        if tipo not in K_TIPO:
            raise ValueError(f"Tipo de entrada '{tipo}' invalido. Opciones: {list(K_TIPO)}")
        total += k_alt(h, beta) * k_dd(dd) * k_tipo(tipo)
    return total


def normalizar_carga(l_sesion: float) -> float:
    return min(l_sesion / L_MAX_REFERENCIA * 100.0, 100.0)
