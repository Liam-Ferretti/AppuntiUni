# Registro aggiornamento appunti

Questo file e `registro_trascrizioni.json` servono per aggiornare in automatico gli appunti LaTeX.

## Procedura (per le prossime volte)
1. Elencare `TrascrizioniAudio/` (ricorsivo, escludendo `venv/`) e confrontare con `file_elaborati` nel JSON: un `.txt` non presente (o con `mtime_ms`/`bytes` diversi) è nuovo.
2. Associare la trascrizione alla materia (cartella `Termodinamica`/`Computazionale` o prefisso del nome, es. `Termo ...`).
3. Aprire il quaderno della materia in `Quaderni/` e leggere solo le pagine dopo `quaderno_ultima_pagina_elaborata`.
4. Aggiungere il contenuto in fondo al relativo `appunti.tex`, dopo `ultima_sezione_tex`, nello stile esistente (ambienti definizione/osservazione/esempio, tikz per le figure, correzioni degli errori segnalate con `osservazione`). Le trascrizioni Whisper sono rumorose: il quaderno è la fonte principale, la trascrizione serve a completare e chiarire.
5. Aggiornare il JSON: nuovi file in `file_elaborati`, nuova `quaderno_ultima_pagina_elaborata`, `ultima_aggiunta`, `ultimo_aggiornamento`.

## Stato attuale (28/10/2026)
- **Termodinamica e Laboratorio**: quaderno elaborato fino a p. 40 (Legge di Dalton). Prossima lezione: calcolo del rendimento di Carnot.
- **Lab. Fisica Computazionale 1**: quaderno elaborato fino a p. 16 (Runge-Kutta RK2/RK4).
