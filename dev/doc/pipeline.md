# Pipeline di generazione fotografica

La pipeline è orchestrata con **LangGraph** e viene eseguita interamente premendo **Generate photo**.

| Step | Cosa fa | Modello / strumento |
|---|---|---|
| **1. Preparazione dell’immagine** | Corregge l’orientamento della foto, converte in RGB e ridimensiona a un massimo di 2048 px. | **Python + Pillow**: elaborazione locale dell’immagine. |
| **2. Identificazione del capo** | Individua il capo centrale con un punto guida; confronta fino a tre maschere e sceglie quella centrale, escludendo componenti periferiche come la manica di chi lo regge. Se manca una maschera centrale valida, interrompe il lavoro. | **`fal-ai/sam-3/image`**: segmentazione guidata da testo e punto centrale. **NumPy + OpenCV**: selezione geometrica della maschera. |
| **3. Correzione dell’illuminazione** | Stima la dominante luminosa da zone dello sfondo potenzialmente neutre e corregge i canali RGB. Prepara la foto corretta e il capo isolato su bianco. | **Python + NumPy + OpenCV**: algoritmo locale, senza chiamate AI. Se i riferimenti sono insufficienti o discordanti, segnala l’incertezza. |
| **4. Descrizione del capo** | Analizza un ritaglio fino a 1280 px, JPEG qualità 95 senza sottocampionamento cromatico, richiedendo al massimo 120 parole su tipo, vestibilità, costruzione, colori e stampe. Ragionamento esteso disattivato, cache per un’ora e limite di sicurezza invariato di 90 secondi. La descrizione è obbligatoria: un errore interrompe il lavoro prima della generazione. | **`fal-ai/moondream3-preview/query`**: analisi visiva con OCR, `reasoning=false`, `temperature=0`. Sostituisce SA2VA 8B; i vecchi modelli SA2VA restano selezionabili per confronto. |
| **5. Generazione fotografica** | Riceve due viste del solo capo selezionato: un ritaglio ravvicinato e un riferimento isolato con l’inquadratura originale, più la descrizione visiva. Genera una foto flat lay da catalogo, con finitura completamente stirata e rimozione delle imperfezioni d’uso, chiedendo di preservare colori e dettagli. | **`fal-ai/nano-banana-pro/edit`**, predefinito a **2K**: modello di generazione ed editing con più immagini di riferimento e istruzioni testuali. |
| **6. Stiratura dedicata** | Modifica soltanto la foto appena generata con un prompt dedicato alla rimozione completa di grinze, pieghe accidentali e segni d’uso, preservando colori, stampe, trama e costruzione. Non riceve nuovamente l’originale spiegazzato. Se fallisce, la pipeline si interrompe. | **`fal-ai/nano-banana-pro/edit`**: seconda chiamata di editing alla stessa risoluzione, senza tentativi automatici aggiuntivi. |
| **7. Rimozione dello sfondo e composizione** | Estrae il capo dalla foto passata attraverso la stiratura, rifinisce i bordi e lo centra su un fondo bianco reale, formato **1200 × 1800 px**. | **`fal-ai/birefnet/v2`**: modello di rimozione dello sfondo con trasparenza. **Pillow** esegue poi la composizione locale. |
| **8. Controllo qualità** | Confronta i colori con il riferimento corretto, controlla separatamente le zone colorate e verifica l’uniformità del bianco. Registra anche metriche diagnostiche di forma e dettaglio. | **Python + OpenCV**: controlli numerici locali. Non verifica l’assenza di pieghe, il significato delle scritte o l’identità esatta delle stampe. |

## Prompt e configurazione

Il prompt fotografico predefinito è in [backend/prompts/generation.txt](../btc-sartiq/backend/prompts/generation.txt). Descrizione visiva e valutazione della luce vengono inserite automaticamente.

Il prompt di stiratura è in [backend/prompts/pressing.txt](../btc-sartiq/backend/prompts/pressing.txt). Entrambi sono riletti a ogni esecuzione e sono consultabili nei campi API `generation_prompt` e `pressing_prompt`. Il passaggio ha progresso, anteprima e traccia di modello/librerie dedicati. La rimozione delle pieghe è un requisito esplicito dei prompt, ma l’assenza effettiva va verificata visivamente: non è certificata dal controllo OpenCV.

Sono **5 inferenze fal per immagine senza cache**: SAM → Moondream → Nano Banana (generazione) → Nano Banana (stiratura) → BiRefNet; diventano **4** se la descrizione è già in cache. La stiratura aggiunge una chiamata e relativo tempo/costo. Le altre operazioni sono locali. È configurabile anche `fal-ai/nano-banana-2/edit` come alternativa al generatore Pro, per entrambi i passaggi. I riferimenti del generatore restano senza perdita e mantengono la risoluzione completa, indipendentemente dal ritaglio per l’analisi.

La latenza di Moondream e la qualità delle descrizioni vanno ancora misurate su fal; code, avvii a freddo o disservizi possono comunque causare timeout. Il vecchio `FAL_ANALYSIS_REQUIRED=false` non consente più di saltare l’analisi. [API ufficiale Moondream](https://fal.ai/models/fal-ai/moondream3-preview/query/api).

La correzione cromatica resta una stima: il recupero esatto dei colori richiede riferimenti neutri affidabili sotto la stessa luce del capo.
