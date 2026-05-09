# Modelo secundario para la gestión de herramientas

Estoy pensando en usar el LLM secundario para herramientas más complejas y multipaso 

De esta manera puedo tener tres tipos de procesamiento en función del tipo de herramientas y la complejidad de la query del usuario.

- Tools de voz: sería principalmente el evaluate_query y tools específicas de voicbot como set_conversation_mode
- Tools rápidas y sencillas o multipaso liegero: como el web_search, capturar screenshot, y otras.
- Tools complejas o multipaso: todas las demás.

Las tools de voz las usuaria e implementaría el LLM de voz principal.
Las tools rápidas y sencillas las implementaría el LLM secundario.
El resto de las tools o peticiones complejas y multipaso, serían redirigidas a Hermes por el LLM secundario.

## Los tres niveles claros

Nivel 1 – Tools de voz (gestionadas directamente por el LLM principal)Ejemplos: evaluate_query, set_conversation_mode, change_voice_style, get_current_context, interrupciones, etc.
Ventaja: Latencia mínima (una sola inferencia del Qwen3.5-35B-A3B).
El modelo principal las maneja de forma nativa, sin saltos extra.
Ideal para todo lo que deba sentirse instantáneo.

Nivel 2 – Tools rápidas y sencillas (gestionadas por el LLM secundario / orquestador)Ejemplos: web_search (simple), capture_screenshot, read_screen_text, play_local_media, recordatorios locales, control básico de dispositivos locales, etc.
El LLM secundario las ejecuta directamente y devuelve el resultado al LLM principal de forma sincrónica.
Latencia aceptable (dos inferencias + ejecución rápida).
Mantienes control local y respuesta casi inmediata.

Nivel 3 – Tools complejas o multipaso (delegadas a Hermes por el LLM secundario)Todo lo demás: web_search profunda + razonamiento, análisis de documentos largos, tareas multi-step, uso de perfiles multi-agent de Hermes, shell complejo, etc.
El LLM secundario detecta esto, lanza la sesión ACP a Hermes de forma asíncrona, y devuelve inmediatamente un mensaje de “estoy en ello” al usuario.
Cuando Hermes termina, tu protocolo actual ya mete el resultado en el contexto del voicebot → LLM principal responde.

Esto es una jerarquía clara:LLM Principal → decide si es Nivel 1 o llama a evaluate_query
LLM Secundario (orquestador) → decide entre Nivel 2 (ejecuta) o Nivel 3 (delegar a Hermes)

Cómo implementarlo sin añadir mucha latencia

Dado que tu Qwen3.5-35B-A3B vuela en la M4 Pro (70+ tokens/s en benchmarks similares), puedes permitirte dos inferencias en muchos casos, siempre que optimices:LLM Principal: Prompt muy estricto para que solo llame evaluate_query cuando sea realmente necesario (nivel 2 o 3). Chitchat y tools de voz → respuesta directa.
LLM Secundario: Hazlo lo más rápido posible. Opciones reales:Misma Qwen3.5-35B-A3B en paralelo (dos instancias).
O baja a Qwen3.5-14B-A3B / 8B quantizado para el orquestador (más rápido aún, y suficiente para clasificar + ejecutar tools simples).

Usa structured output (JSON mode) obligatorio en ambos para minimizar parsing y alucinaciones.

Prompts recomendados (versión lista para copiar)System prompt LLM Principal (Voicebot)

Eres Jarvis VoiceBot: asistente de voz ligero, natural y ultra-rápido.
- Mantén siempre un tono conversacional y respuestas cortas para voz.
- Tienes tools de voz internas (como set_conversation_mode, etc.). Úsalas directamente cuando corresponda.
- Solo tienes una tool externa: evaluate_query. Úsala ÚNICAMENTE cuando la petición del usuario requiera búsqueda, captura de pantalla, acciones del sistema o cualquier procesamiento que no sea charla pura o tool de voz simple.
- Para todo lo demás responde directamente sin tools.

Tool definition para evaluate_query (en el principal)json

{
  "name": "evaluate_query",
  "description": "Envía la consulta del usuario al orquestador de herramientas para que decida si puede ejecutarla rápidamente o delegarla a Hermes.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_query": { "type": "string", "description": "La consulta exacta del usuario, tal cual." }
    },
    "required": ["user_query"]
  }
}

System prompt LLM Secundario (Orquestador)

Eres el Orquestador de Herramientas de Jarvis VoiceBot. Actúa de forma rápida y determinista.

Clasifica la user_query en uno de estos tres niveles:
- Nivel 2 (rápido): Ejecuta directamente tools sencillas como web_search simple, capture_screenshot, etc. Devuelve el resultado.
- Nivel 3 (complejo): Inicia sesión ACP a Hermes con una descripción clara de la tarea. Devuelve inmediatamente un mensaje para el usuario indicando que se ha iniciado la tarea.

Nunca ejecutes tools complejas o multipaso tú mismo.
Devuelve siempre JSON estructurado con:
{
  "level": 2 | 3,
  "action": "execute" | "delegate",
  "result": "..." (solo si level 2),
  "hermes_task": "descripción clara para Hermes" (solo si level 3),
  "user_message": "mensaje natural que dirá el voicebot al usuario ahora mismo"
}

## Additional info

❯ Aún así creo que el uso del llm secundario podría ser útil en ocasiones. Por ejemplo, este seundario podría tener el  
thinking activado y ser el que comunique con el agente externo; de esta manera quizás pudiéramos mejorar el voicebot,   
sin añadir latencia al llm principal. ¿Qué te parece? ¿Qué utilidades o funcionalidades añadirías al LLM secundario?    
                                                                                                                        
⏺ La idea tiene mucho sentido. El punto clave es la asincronía: si el secundario trabaja mientras el primario ya ha     
  respondido "en ello", el thinking del secundario no añade latencia perceptible al usuario.                            
                                                                                                                        
  Lo que el secundario con thinking aportaría que el primario no puede hacer bien:                                      
                                                                                                                        
  1. Formulación de tareas para Hermes                                                                                  
  El primario recibe la petición en voz natural ("busca información sobre X y dime qué me recomiendas") y la pasa cruda 
  a run_agent. El secundario con thinking podría reformularla en un prompt preciso y estructurado para Hermes — mejor   
  calidad de delegación sin coste para el primario.                                                                     
                                                                                                                        
  2. Síntesis de resultados de herramientas                                                                           
  web_search devuelve JSON crudo de SearXNG. Ahora mismo el primario tiene que procesar eso en su segunda inferencia. El
   secundario podría filtrar, resumir y extraer lo relevante antes de devolvérselo al primario — el primario recibe una 
  respuesta limpia, no datos crudos.                                                                                    
                                                                                                                        
  3. Cadenas multi-paso sin contaminar el contexto primario                                                           
  web_search → leer resultado → refinar búsqueda → sintetizar — actualmente cada paso requiere una inferencia primaria. 
  El secundario podría ejecutar toda la cadena y devolver solo el resultado final.                                      
                                                                                                                        
  4. Lo que ya hace, pero mejor con thinking:                                                                           
  - Consolidación y memoria: ya lo hace en background, thinking mejoraría calidad de extracción                         
  - Screenshot + análisis: ya usa el secundario, thinking mejoraría el razonamiento sobre la imagen                     
                                                                                                                        
  ---                                                                                                                   
  El patrón que propongo:                                                                                               
                                                                                                                        
  Primario detecta tarea compleja                                                                                       
    → responde al usuario: "Dame un segundo..."                                                                         
    → lanza secundario async con thinking=true                                                                          
        ↓                                                                                                               
    Secundario (thinking) planifica, ejecuta herramientas, habla con Hermes si hace falta                               
        ↓                                                                                                               
    Resultado inyectado en contexto del primario                                                                        
    Primario responde con la información ya procesada                                                                   
                                                                                                                        
  La diferencia clave vs. la arquitectura de tres niveles que descartamos: el secundario NO es un router ni un          
  orquestador que se interpone en el flujo normal. Solo entra para trabajo profundo que ya es async por naturaleza. El  
  primario sigue siendo el único que habla con el usuario.                                                              
                                                                                                                      
  ---
  Lo que NO daría valor:                                                                                                
  - Poner el secundario en el camino crítico de tools rápidas (web_search simple, clipboard) — añadiría latencia sin    
  ganancia                                                                                                              
  - Usarlo como verificador de respuestas del primario — el timing no encaja con el flujo de voz                        
                                                                                                                        
  ¿Quieres que esto sea el próximo paso de implementación, o primero resolvemos el problema con Hermes?  