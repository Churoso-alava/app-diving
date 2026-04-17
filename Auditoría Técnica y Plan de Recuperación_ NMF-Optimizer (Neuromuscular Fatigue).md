# **Auditoría Técnica y Plan de Recuperación: NMF-Optimizer (Neuromuscular Fatigue) v4.4**

El sistema NMF-Optimizer (Neuromuscular Fatigue Optimizer) v4.4 es una plataforma especializada en el análisis de carga y fatiga para atletas de buceo, centrada en métricas de rendimiento neuromuscular y riesgo. Tras una verificación directa del repositorio, se ha determinado que el núcleo algorítmico no se basa en factorización de matrices, sino en un **Sistema de Inferencia Difusa tipo Mamdani** (vía skfuzzy) y en el cálculo de métricas de carga de trabajo como el **ACWR (Acute:Chronic Workload Ratio)**.

## **1\. Auditoría de Reestructuración de Archivos Críticos**

Esta tabla corrige las definiciones de funciones y archivos para cumplir con los requisitos de la suite de pruebas (pytest).

| Jerarquía | Archivo | Errores Detectados | Acción Correctiva | Requisito de Test (AST/Literal) |
| :---- | :---- | :---- | :---- | :---- |
| **1\. Datos** | db.py (Raíz) | **Ausente**. | Crear shim que re-exporte de data/db.py. | Obligatorio para test\_audit\_fixes.py. |
| **2\. Lógica** | logic/services.py | Importación fallida. | Corregir casing a snake\_case (ej. vmp\_hoy). | Los tests fallan si detectan PascalCase. |
| **3\. Motor** | fuzzy/fuzzy\_engine.py | Dependencia faltante. | Instalar scikit-fuzzy. | Motor Mamdani v4.1 (23 reglas). |
| **4\. UI** | components/tab\_dashboard.py | Nombre incorrecto. | Cambiar a def tab\_dashboard(df):. | El test busca este nombre exacto. |
| **5\. Core** | app.py (Raíz) | **Ausente**. | Crear archivo con 14 checks de estructura. | Debe contener @st.cache\_data literal. |

## ---

**2\. Diagnóstico Técnico de Bloqueadores (Baseline)**

La ejecución actual de la suite de pruebas muestra un estado crítico que impide el desarrollo del motor Mamdani:

* **Errores de Colección:** 4 fallos críticos impiden que 19 tests se ejecuten. La ausencia de app.py en la raíz es el principal bloqueador.  
* **Falsos Positivos de Arquitectura:** El sistema no utiliza Factorización de Matrices No Negativas. El uso de "NMF" en el nombre se refiere estrictamente a **Neuromuscular Fatigue**. Las fórmulas de reducción de dimensionalidad son irrelevantes para este repositorio.  
* **Fallo de Inferencia Difusa:** Los tests de test\_fuzzy\_variables.py crashean por la falta del módulo skfuzzy en el entorno de ejecución.

## ---

**3\. Especificaciones para la Recuperación del Backend**

Para que los 280 registros de Supabase fluyan correctamente hacia el tablero, se deben aplicar los siguientes cambios técnicos:

### **3.1. Implementación de @st.cache\_data (Validación AST)**

Los tests de auditoría verifican el Árbol de Sintaxis Abstracta (AST) de Python. No basta con que el código funcione en runtime; el decorador debe estar escrito literalmente en el código para que el test test\_audit\_fixes.py lo valide.

Python

\# CORRECTO (Pasa el test AST)  
@st.cache\_data(ttl=30)  
def calcular\_historial\_batch\_cached(df):  
   ...

\# INCORRECTO (Falla el test aunque funcione)  
\_cache\_data\_ttl \= st.cache\_data(ttl=30)  
@\_cache\_data\_ttl  
def calcular\_historial\_batch\_cached(df):  
   ...

### **3.2. Estructura de Paquetes**

Es imperativo añadir archivos \_\_init\_\_.py faltantes, especialmente en utils/, para evitar avisos de paquetes incompletos que detienen la recolección de pytest.

## ---

**4\. Plan de Acción de 5 Fases (Actualizado T1-T8)**

Este plan sigue el orden lógico para restaurar la visibilidad del backend y cumplir con los 14 checks específicos de la suite de pruebas.

### **Fase 1: Shims de Raíz y Estructura (T1-T2)**

* Crear db.py en la raíz: from data.db import \*.  
* Crear utils/\_\_init\_\_.py para sanear el árbol de módulos.

### **Fase 2: Entorno y Dependencias (T3)**

* Actualizar requirements.txt con versiones fijas (**Pinning**):  
  * scikit-fuzzy==0.4.2  
  * streamlit==1.38.0  
  * pillow==10.3.0  
  * pandas==2.1.4.

### **Fase 3: Corrección de Servicios y Casing (T4-T5)**

* Ajustar logic/services.py para manejar el contrato de datos snake\_case proveniente de Supabase.  
* Corregir test\_diving\_load.py para que importe correctamente desde logic.biomechanics.

### **Fase 4: Construcción del Orquestador app.py (T6-T7)**

* Crear el punto de entrada app.py en la raíz (no main\_app.py).  
* Definir físicamente tab\_dashboard(df) y aplicar los decoradores de caché literales necesarios para superar los tests de seguridad y UI.

### **Fase 5: Verificación y Motor Mamdani (T8)**

* Ejecutar pytest. Se espera pasar de 17 tests a más de 65 tests exitosos.  
* Una vez estabilizada la base, proceder con la expansión de las 23 reglas del motor de lógica difusa en fuzzy/fuzzy\_engine.py.

## **Conclusión**

El informe de auditoría previo contenía errores factuales sobre la naturaleza científica del proyecto. Al corregir la interpretación de **NMF** como **Neuromuscular Fatigue** y enfocar los esfuerzos en la creación de los archivos de raíz (app.py, db.py) y la instalación de skfuzzy, se garantiza que el flujo de datos desde la capa de persistencia hasta el motor de inferencia sea estable y verificable por la suite de pruebas actual.