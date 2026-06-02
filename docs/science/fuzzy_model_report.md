# Informe: Fundamentos Fisiológicos del Modelo de Fatiga

## 1. Introducción
Este modelo de inferencia difusa ha sido diseñado como una herramienta de apoyo a la decisión clínica y deportiva, fundamentado en la monitorización de la carga interna y el estado neuromuscular del atleta. Su objetivo no es la computación estadística per se, sino la **traducción de múltiples indicadores fisiológicos en un estado funcional** que permita ajustar la carga de entrenamiento en tiempo real.

## 2. Justificación Fisiológica de las Variables
La selección de las variables antecedentes responde a una necesidad de monitoreo holístico de los sistemas afectados por la carga de entrenamiento:

*   **VMP (Velocidad Media Propulsiva - `vmp_hoy`):** Indicador directo del estado de fatiga del Sistema Nervioso Central (SNC) y la capacidad de reclutamiento de unidades motoras. Es la variable principal de "preparación neuromuscular".
*   **ACWR (Acute:Chronic Workload Ratio - `acwr`):** Refleja el equilibrio entre el estímulo agudo (fatiga acumulada reciente) y la capacidad de carga crónica (adaptación).
*   **Delta Porcentual (`delta_pct`):** Evalúa la respuesta adaptativa inmediata al último estímulo. Permite detectar si el atleta se encuentra en un proceso de recuperación o de estrés agudo.
*   **Z-Meso (`z_meso`):** Indicador de la adaptación a largo plazo. Ayuda a distinguir entre una fatiga funcional normal y un estancamiento en el rendimiento.
*   **Beta Aguda y Beta 28 (`beta_aguda`, `beta_28`):** Derivadas de las tendencias del VMP. Permiten monitorizar la cinemática de la fatiga, diferenciando entre fluctuaciones diarias y tendencias preocupantes de deterioro neuromuscular.
*   **Wellness y Carga Integrada (`wellness_norm`, `carga_integrada_plan`):** Variables de contexto que modulan la respuesta fisiológica, permitiendo interpretar si el estrés del atleta es puramente físico o está exacerbado por factores externos (psicosociales, recuperación insuficiente).

## 3. Lógica Fisiológica de las Reglas (Toma de Decisiones)
Las reglas no son arbitrarias; representan la lógica fisiológica de gestión de la fatiga:

*   **Estado Crítico (Neuromuscular):** Se dispara cuando el VMP cae drásticamente (fatiga SNC) y existe un indicador de estrés agudo (Alarma de Delta o Z-Meso negativo). Fisiológicamente, esto representa una **incapacidad del sistema neuromuscular para generar fuerza**, elevando exponencialmente el riesgo de lesión y sobreentrenamiento.
*   **Fatiga Acumulada (Sistémica):** La combinación de una ACWR inadecuado (bajo o excesivo) junto con tendencias negativas en el VMP (`beta_aguda` negativa, `deterioro` en `beta_28`) indica un **fracaso en el proceso de supercompensación**. El sistema de reglas detecta esta incapacidad sistémica para absorber la carga planificada.
*   **Alerta Temprana (Ajuste Preventivo):** Reglas que detectan desviaciones leves en `delta_pct` o `z_meso` junto con un `wellness` moderado actúan como un sistema de **"freno preventivo"**. Su objetivo fisiológico es evitar que el atleta alcance estados de sobreesfuerzo, reduciendo la intensidad antes de que la fatiga se vuelva sistémica.
*   **Estado Óptimo (Adaptación Positiva):** Reglas que validan una ACWR en rango, `delta_pct` en zona de ganancia y `wellness` alto. Fisiológicamente, esto indica una **asimilación eficiente del entrenamiento** y una reserva funcional alta.

## 4. Conclusión
La toma de decisiones de este motor difuso emula el razonamiento del fisiólogo deportivo: prioriza la integridad del sistema neuromuscular (VMP), contextualiza la carga externa (ACWR/Plan) y modula la respuesta según la recuperación subjetiva (Wellness). El modelo no busca reducir al atleta a un número, sino representar la complejidad de su respuesta adaptativa al estrés físico.

## 5. Limitaciones Estadísticas y Robustez
Para mantener la validez fisiológica del modelo, se asume que las métricas calculadas son estadísticamente robustas. Se reconoce que métricas basadas en distribuciones normales (como `z_meso`) requieren validación estadística previa en los datos individuales de cada atleta (Test de Shapiro-Wilk). Si la distribución de datos no es normal, el modelo debe aplicar técnicas de robustez estadística (como el uso de percentiles o factores de atenuación) para evitar interpretaciones fisiológicas erróneas derivadas de sesgos estadísticos.
