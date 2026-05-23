pub fn get_notification(key: &str, lang: &str) -> &'static str {
    match (key, lang) {
        ("startup", "es") => {
            "[Sistema: el voicebot acaba de arrancar. Son las {time_str}, del día {date_str}\n Saluda al usuario de forma natural y muy concisa.]"
        }
        ("startup", "en") => {
            "[System: the voicebot just started. It's {time_str} on {date_str}\n Greet the user naturally and briefly.]"
        }

        ("background_task_done", "es") => {
            "[Sistema: una tarea en segundo plano ha terminado.]\n Tarea: {task}\n Resultado: {result}\n Informa al usuario de forma natural y concisa."
        }
        ("background_task_done", "en") => {
            "[System: a background task has finished.]\n Task: {task}\n Result: {result}\n Inform the user naturally and briefly."
        }

        ("acp_permission", "es") => {
            "[Sistema: el agente ACP necesita permiso para realizar una acción.]\n Acción solicitada: {question}\n Opciones: {opts_str}\n Pregunta al usuario de forma natural si desea permitirlo (sí/no)."
        }
        ("acp_permission", "en") => {
            "[System: the ACP agent needs permission to perform an action.]\n Requested action: {question}\n Options: {opts_str}\n Ask the user naturally if they want to allow it (yes/no)."
        }

        ("reorganize_memory", "es") => {
            "[Sistema: necesitas reorganizar tu memoria para seguir conversando. Avisa al usuario de que vuelves en unos minutos.]"
        }
        ("reorganize_memory", "en") => {
            "[System: you need to reorganize your memory to keep conversing. Tell the user you'll be back in a few minutes.]"
        }

        ("memory_reorganized", "es") => {
            "[Sistema: has terminado de reorganizar tu memoria. Son las {now}. Avisa al usuario de que ya estás disponible de nuevo.]"
        }
        ("memory_reorganized", "en") => {
            "[System: you've finished reorganizing your memory. It's {now}. Tell the user you're available again.]"
        }

        _ => "[Sistema: el voicebot acaba de arrancar.]\n Saluda al usuario.",
    }
}
