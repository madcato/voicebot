# Voicebot: process architecture

This file explains the basic process architecture of the application. It defines what thread must be always running and its responsabilities. Also defines the signals aned evetns that threads interchange to intercommunication and collaboration.

IMPORTANT: The speaker and microphone of this system has hardware echo cancelation.

## Session object

- The session object is shared almost by every thread. 
- Threads must not block it for long time, only fast reads and writes.

### Required properties

- **transliterated_text**: The text already transliterated by the STT
- **assitant_text**: The assistant text (from LLM) no yet played.
- **sentences** Array of sentenes to be played by the TTS.
- **llm_post_finished**: A boolean value indicating if the llm POST has finished. Default false.

## Threads
- Main
- STT
- VAD
<!-- - SPV -->
- LLM
- SEN
- TTS
- SUM

### Definitions of each thread responsability

#### Main

- Initializing all the shared objects.
- Launches all the threads.
- Keep listening to the system signals, like Ctrl + C, to handle them.

#### STT - Speech-To-Text

- This thread receives the audio input.
- This thread realizes the Speech-To-Text proccess with the configured STT provider.
- The transcribed text is stored in the session object.
- This thread is always running and always transcribing.
- It launches the signal **VAD_DETECTED** every time text is transliterated. This cancel all LLM previous processing, and current assitant voice and clears the remaining assistant text no yet played.


#### VAD - Voice-Activity-Detection

- This thread receives the audio input.
- It launches the event **VAD_FINISH** when the voice is absent.
- It launches the event **VAD_FINISH** in the minimal period needed by the VAD provider.
- *VAD_SILENCE_MS* environment variable defines the minimal miliseconds to detect voice non activity.

<!-- #### SPV - Speaker-Verification

- This thread uses the speaker verification provider to detect the main user voice.
- When detects a secondary voice, it launches the signal **SECONDARY_VOICE_DETECTED**
- (This behaviour must not be implemented right now)
- This thread must run only when signal **VAD_DETECTED** appear. -->

#### LLM - Large-Language-Model client

- It has the llm client to send the messages to the LLM provider.
- Each time the event **VAD_FINISH** is received, stores the text to be sent in the chat history and cleans the session transliterated text. This way everything the user says is kept, even it is not sent because the user continues speaking. Maybe this could make that to "user" role messages or more are sent at the same time, but this is not a problem for LLM.
- After the user text is stored in the chat history is sent to the llm provider.
- When this thread is sending o receiving the POST, it doesn't checks the **VAD_FINISH** event, this event only wakes up it.
- This means the thread must blocked until a **VAD_FINISH** arrive.
- When the signal **VAD_DETECTED** is received, the thread must stop everything is doing at any state: close the connection with LLM provider and return to be blocked until **VAD_FINISH**.
- When any text is received by streaming, it is stored in the session and the event **LLM_POST_RECEIVED** is sent.
- When the POST finished, on the final chunk, set llm_post_finished = true, then fire one last LLM_POST_RECEIVED to wake SEN,
  then fire LLM_POST_FINISHED. The wording should be clarified to say this only applies to the final LLM_POST_RECEIVED.

#### - SEN - Sentence Splitter

- When **LLM_POST_RECEIVED** is received, it check if there is a complete sentence to be transliterated.
- When the signal **VAD_DETECTED** is received, the thread must stop everything is doing at any state: clean the session received text and return to be blocked until next **LLM_POST_RECEIVED**.
- When a sentence is ready to be transliterated, it must be stored in the session.
- When a sentence is stored it must launch **SENTENCE_READY** event.
- While there is no more text to be splitted, this thread must be blocked until **LLM_POST_RECEIVED** arrive.
- If session llm_post_finished property is true, all the sentences must be sent to the TTS one after another. 

#### TTS - Text-To-Speech

- When the signal **VAD_DETECTED** is received, the thread must stop everything is doing at any state: clean the session received assitant text and remaining sentences.
- This thread is blocked until next **SENTENCE_READY**.
- Text from LLM must be stored in the chat history before is played.
- Transcribe sentence, remove from the sentences session property and play it.

#### SUM

- This thread handles the summarization and user profile extraction process, only when conditions are met.
- It is blocked until the event **LLM_POST_FINISHED** arrives. This way, the process is launched when the user is listeing to the assitant, so the GPU is free.
- Must be cancelled by the **VAD_DETECTED** signal.

## Signals and events

Signals are different from events.

- A signals must be managed instanly, used to cancel other thread current process.
- Events wakes up other threads. The launhing thread must store the data to be processed in shared place; when the receiving thread is blocked, the event wakes up it. Before get blocked, the receiving thread must check there is no data to be process before being blocked. This means that the sending event thread, must store the data, before launching the event.

### Signals

- **VAD_DETECTED**: Every thread must stop its processing when this signal is launched.

### Events

- **VAD_FINISH**: Launched when VAD detects silence.
- **LLM_POST_RECEIVED**: Launched when any text arrives from LLM.
- **SENTENCE_READY**: Launched when any text is ready to be transliterated.
- **LLM_POST_FINISHED**: The LLM has finished sending the response.

#### Special

- There is no need of TTS_DONE event, because this voicbot is always listening, there is no user turn and assistant turn. The user always has the preference, assistant is seondary.