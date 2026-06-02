# Diseño: Restauración de Wellness (Registro Grupal y Escala Visual)

## 1. Objetivo
Restaurar la funcionalidad de registro grupal de Wellness en AppDivingCodex y asegurar la visibilidad de la escala de referencia visual (emojis).

## 2. Componentes y Arquitectura

### 2.1. Escala Visual de Emojis
Se reutilizará el componente existente `components/wellness_legend.py`. Este será renderizado en la parte superior del formulario de registro (tanto individual como grupal) para servir como guía visual continua al staff.

### 2.2. Registro Grupal
Se modificará `components/tab_wellness_registro.py` para incluir un selector de modo (Individual/Grupal).

*   **Estado Grupal:**
    *   Se obtendrá la lista de atletas activos mediante `cargar_atletas()`.
    *   Se construirá un `pandas.DataFrame` con columnas: Atleta, Sueño, Fatiga, Estrés, Dolor, Humor.
    *   Se utilizará `streamlit.data_editor` para la entrada de datos numérica (rango 1-7).
    *   Se implementará un botón de guardado que procese las filas del dataframe y llame a la lógica de inserción de forma masiva (o iterativa si el servicio de DB lo requiere).

## 3. Flujo de Datos
1.  Staff selecciona modo "Grupal".
2.  Sistema renderiza tabla con valores por defecto (ej. 4) para todos los atletas.
3.  Staff ajusta valores numéricos en la tabla.
4.  Staff presiona "Guardar".
5.  El componente recorre el DataFrame y llama a `insertar_wellness` por cada atleta con datos ingresados.

## 4. Validación
*   Se validará que los valores estén en el rango [1, 7].
*   Se manejarán errores de inserción de forma grupal informando cuáles registros fallaron (si aplica).

## 5. Próximos pasos
1. Aprobar este diseño.
2. Invocar skill de `writing-plans` para implementar los cambios.
