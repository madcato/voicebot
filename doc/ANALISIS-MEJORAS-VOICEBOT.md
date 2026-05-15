# Análisis de Mejoras y Nuevas Funcionalidades para Voicebot

**Fecha:** 5 de Abril de 2026  
**Proyecto:** Voicebot (Rust)  
**Referencia:** [Voicebot Article](https://x.com/danieltvela/status/2033972678197997743)

---

## Resumen Ejecutivo

El proyecto Voicebot ya cuenta con una arquitectura sólida y bien diseñada que cubre la mayoría de las funcionalidades descritas en el artículo de referencia. La base del pipeline STT→LLM→TTS está bien implementada, junto con integración de herramientas, memoria persistente, monitoreo de pantalla (EYES), y comunicación asíncrona con agentes.

A continuación se detallan las mejoras y funcionalidades adicionales categorizadas por prioridad.

---

## 1. Funcionalidades Requeridas (Core - Alta Prioridad)

### 1.1 Optimización de Latencia End-to-End

**Estado actual:** Usa mlx-lm/oMLX con KV-cache implícito y streaming parcial de Whisper.

**Mejoras propuestas:**

- **Continuous Streaming en Whisper**: En lugar de procesar chunks de audio completos, implementar streaming continuo de transcripción para reducir latencia de primer token.
- **Pre-carga de modelo LLM**: Mantener el modelo en memoria activa entre turnos para evitar tiempo de carga en cada interacción.
- **VAD adaptativo**: Reducir el umbral de silence de 1.5s a valores configurables (ej: 800ms) para respuestas más rápidas.
- **Early termination en TTS**: Sintetizar la primera frase mientras el LLM genera las siguientes (ya implementado parcialmente con SentenceSplitter).

**Impacto:** Crítico para la experiencia de usuario — la velocidad es una de las características clave de un voicebot.

### 1.2 Detección de Intención de Conversación (Conversation Awareness)

**Estado actual:** Tiene wake word configurable y speaker verification.

**Falta implementar:**

- Detectar cuando el usuario está hablando **al voicebot** vs. speaking-elsewhere (conversación con otras personas o hacia la pantalla).
- Diferenciar entre comandos directos ("Hey Voicebot, haz esto") vs. respuestas contextuales.
- Soporte para múltiples usuarios con identificación de quién habla.

**Propuesta técnica:**

- Usar el speaker embedding existente para identificar al hablante.
- Analizar el contexto de la transcripción para determinar si es un comando dirigido al bot.
- Threshold configurable de confianza para activar respuesta.

### 1.3 Fallback y Manejo de Errores Robusto

**Estado actual:**_ERR_ en logging básico.

**Falta:**

- Reconocimiento automático cuando el STT no entiende nada (silencio, ruido, idioma incorrecto).
- Reintentos automáticos con estrategias diferentes (distinto modelo, distinto prompt de idioma).
- Feedback auditivo claro cuando no se entiende (en lugar de silencio total o error).
- Modo degradado: si el LLM falla, responder con TTS indicando el problema.

**Impacto:** Esencial para uso real — un bot que guarda silencio cuando no entiende frustra al usuario.

### 1.4 Mode Switching Inteligente

**Estado actual:** Estados básico (Active/Ambient) con wake word.

**Mejoras:**

- Transiciones fluidas entre modos:
  - **Passive**: Solo escucha wake word
  - **Active**: Respondiendo a usuario
  - **Background**: ProcesoDaemon activo (ej: buscando info)
  - **Alert**: Notificando proactivamente
- Indicadores sonoros/visuales del cambio de modo (opcionales, configurables).

---

## 2. Funcionalidades Recomendadas (Alta Prioridad)

### 2.1 Herramientas Expandidas

| Herramienta | Descripción | Prioridad |
|------------|-------------|-----------|
| **Calendario** | Consultar eventos próximos, crear recordatorios | Alta |
| **Notas** | Crear/leer notas en sistema o servicios externos (Obsidian, Notes) | Alta |
| **Control de dispositivos** | Integración con OpenHue (luces), Home Assistant | Media |
| **Weather** | Consultar clima actual/pronóstico | Media |
| **Email** | Leer/enviar emails básicos (IMAP/SMTP) | Media |
| **Notificaciones** | Leer notificaciones del sistema | Baja |

### 2.2 Contexto Visual Avanzado (EYES++)

**Estado actual:** Captura de pantalla cada N segundos, análisis básico.

**Mejoras recomendadas:**

- **Detección de contenido sensible**: Alertar si hay passwords visibles, datos bancarios, información personal en pantalla.
- **OCR integrado**: Extraer texto de la pantalla para contexto del LLM (no solo análisis visual).
- **Comparación de frames**: Detectar cambios significativos vs. pantalla estática para evitar alertas innecesarias.
- **Historial de estados**: Recordar qué ventanas tenía abiertas el usuario para notar cambios.

### 2.3 Integración con Ecosistema de Agentes

**Estado actual:** Integración básica con Hermes Agent via ACP.

**Mejoras:**

- **Fallback entre agentes**: Si Hermes falla, intentar otro agente disponible.
- **Selección dinámica de agente**: Berdasarkan el tipo de tarea (codificación vs. búsqueda vs. análisis).
- **Comunicación bidireccional**: El agente puede pedir clarificación al usuario (el voicebot habla: "¿continúo?").
- **Timeout configurable por tipo de tarea**: Mayor para tareas complejas, menor para queries simples.

### 2.4 Conversaciones Proactivas (Initiative Conversations)

**Estado actual:** Daemon básico implementado (cada 5 min configurable).

**Funcionalidades adicionales:**

- Recordatorios proactivos: "Tienes una reunión en 15 minutos"
- Información contextual basada en tiempo/ubicación: "Buenos días, el tráfico está fluide"
- Sugerencias basadas en patrones del usuario: "Suele revisar su email a esta hora"
- Alertas de EYES configurable: "Llevas 2 horas en la misma ventana, ¿quieres un descanso?"

### 2.5 Control de Contexto y Memoria a Largo Plazo

**Estado actual:** Sistema de memorias en SQLite, resumen automático.

**Mejoras:**

- **Preferencias del usuario aprendidas**: Recordar cómo le gusta ser atendido (velocidad TTS, formality, etc.)
- **Historial de tareas**: Qué comandos ha dado el usuario anteriormente para predicción
- **Base de conocimientos personal**: Hechos sobre el usuario, proyectos, contactos
- **Memory consolidation activa**: En lugar de solo resumir, reorganizarmemories en categoríasactionable

---

## 3. Funcionalidades Secundarias (Nice to Have)

### 3.1 Multimodalidad Avanzada

- **TTS con emociones**: Variar tono/velocidad según el contenido (noticias vs. chiste vs. alerta)
- **Voz personalizada**: Finetuning de voz para sonar más personal
- **Expresiones faciales (si hay cámara)**: Avatar visual que habla

### 3.2 Integraciones de Plataforma

| Plataforma | Funcionalidad |
|------------|---------------|
| **Home Assistant** | Control de dispositivos, automatizaciones |
| **Philips Hue** | Control de luces (ya hay openhue) |
| **Obsidian** | Notas y conocimiento personal |
| **GitHub** | Notificaciones de PRs, issues |
| **Calendar** | Eventos, reuniones |
| **Email** | Resumen, envío simple |

### 3.3 Modo Especialista / Rutas de Conversación

- **Modo código**: Mayor contexto técnico, herramientas de terminal
- **Modo búsqueda**: Búsqueda web más agresiva
- **Modo privacidad**: No guardar transcripciones, no usar cloud
- **Modo demostración**: Respuestas más cortas, más feedback auditivo

### 3.4 Logging y Métricas

- **Dashboard de métricas**: Latencia promedio, comandos por sesión, errores
- **Exportar conversaciones**: Para análisis o entrenamiento
- **A/B testing de prompts**: Probar diferentes system prompts

### 3.5 Seguridad y Privacidad

- **Encriptación de BD**: Proteger conversaciones pasadas
- **Voice print enrollment avanzado**: Verificación de speaker más robusta
- **Palabras de seguridad**: "Emergency stop" para detener inmediatamente
- **Modo privacidad**: No guardar nada, solo procesamiento en memoria

---

## 4. Comparativa con el Artículo de Referencia

| Feature del Artículo | Estado Actual | Gap |
|---------------------|---------------|-----|
| Conversation Awareness | Parcial (wake word + speaker verification) | Completar detección de intención |
| Superfast Responses | Parcial (cache_prompt + streaming) | Optimizar más |
| Integrated Ecosystem | ✓ Hermes Agent + herramientas | Expandir agentes |
| Built-in Tools | ✓ 10+ herramientas | Añadir más |
| Screen Awareness | ✓ EYES implementado | Mejorar con OCR |
| Bidirectional & Async | ✓ ACP + Daemon | Expandir casos de uso |
| Initiative Conversations | ✓ Daemon básico | Completar funcionalidades proactivas |
| NLUI | ✓ Base implementada | Refinar contexto |

---

## 5. Hoja de Ruta Sugerida

### Fase 1: Estabilidad y Velocidad (Semanas 1-2)

1. Optimizar latencia VAD y TTS
2. Mejorar manejo de errores de speech
3. Detección de intención de conversación

### Fase 2: Expansión de Herramientas (Semanas 3-4)

1. Añadir calendario y notas
2. Integración con Home Assistant
3. Mejorar EYES con OCR

### Fase 3: Inteligencia Contextual (Semanas 5-6)

1. Mejoras en conversaciones proactivas
2. Memoria a largo plazo más rica
3. Selección dinámica de agentes

### Fase 4: polish y Extras (Semanas 7+)

1. TTS emocional
2. Modos specialist
3. Métricas y dashboard

---

## 6. Notas Técnicas

- El proyecto ya usa buenas prácticas: tokio async, channels, SQLite, tracing
- Considerar lazy initialization de modelos para start-up más rápido
- Los thresholds de VAD deben ser ajustables por el usuario (no todos los entornos son iguales)
- La calidad del TTS (Kokoro) es mejor que say/avspeech -> promover como default
- La verificación de hablante con sherpa-onnx es un diferenciador fuerte

---

_Documento generado como análisis de oportunidades de mejora para Voicebot._
