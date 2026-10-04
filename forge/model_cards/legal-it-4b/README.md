---
license: apache-2.0
language:
- it
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: finetune
pipeline_tag: text-generation
library_name: gguf
tags:
- legal
- italian-law
- diritto
- rag
- open-book
- gguf
- llama.cpp
- eullm
---

# legal-it-4b

**Un modello da 4 miliardi di parametri che risponde a domande sul diritto italiano leggendo i testi di legge che gli vengono forniti.** Gira in locale: il file Q4_K_M pesa 2,5 GB e funziona con llama.cpp, Ollama e l'[EULLM Engine](https://github.com/eullm/eullm).

Sulle 429 domande degli esami riservati, mai viste in addestramento, risponde correttamente all'**85,1%**, contro il 77,6% del modello di partenza con lo stesso retrieval. Per la massima qualità c'è [legal-it-8b](https://huggingface.co/eullm/legal-it-8b) (89,0%), che richiede circa il doppio della memoria.

> **Avvertenza.** legal-it-4b è uno strumento di consultazione dei testi normativi, non un parere legale. Può sbagliare, anche con sicurezza. Ogni risposta va verificata sul testo ufficiale (Normattiva, Gazzetta Ufficiale) e, per qualsiasi decisione, con un professionista.

## Cos'è e come va usato

legal-it-4b è addestrato per rispondere **a libro aperto**: riceve la domanda insieme agli articoli di legge pertinenti e risponde basandosi su quelli, citandoli. A libro chiuso, cioè senza testi nel prompt, nessun modello di queste dimensioni risponde in modo affidabile su termini e contenuti degli articoli (meno del 4% delle risposte giuste nelle nostre prove), e questo non fa eccezione.

Il modello va quindi usato con un sistema di retrieval sopra la raccolta normativa. Il prompt è un solo turno utente, senza prompt di sistema, in questo formato:

```text
Testi normativi di riferimento:

[1] codice civile, art. 1385
<testo dell'articolo>

[2] codice civile, art. 1386
<testo dell'articolo>

[3] codice civile, art. 1382
<testo dell'articolo>

Rispondi alla domanda basandoti sui testi sopra, se sono pertinenti.

Domanda: Che cosa succede alla caparra confirmatoria se la parte che l'ha versata è inadempiente?
```

Ogni testo è troncato a 3.000 caratteri, con ` […]` in fondo se è stato tagliato. Se la domanda cita un articolo che la raccolta non contiene, il prompt si apre con una nota, per esempio `Nota: nella raccolta normativa non è presente art. 9999 (codice civile).`, seguita da una riga vuota. In quel caso il modello è addestrato a dire che l'articolo non c'è invece di descriverne un altro.

Le risposte valutate qui sono state generate in modo **greedy** (temperatura 0), con al massimo 400 token. Le risposte sono brevi, in media circa 460 caratteri.

### Retrieval consigliato

I numeri di questa pagina sono ottenuti con questo retrieval, che pesa in tutto circa 1,2 GB e gira anche su un portatile:

1. se la domanda nomina un articolo, quell'articolo viene messo per primo, cercandolo per numero;
2. BM25 e [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), fusi con Reciprocal Rank Fusion (k = 60), sui 50 migliori di ciascuno;
3. [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) riordina i primi 20;
4. i primi 3 testi vanno nel prompt.

L'implementazione è nel modulo `eullm_forge.eval.dense` di [eullm-forge](https://github.com/eullm/eullm/tree/main/forge). La raccolta normativa si costruisce dai pacchetti AKN OpenData di [Normattiva](https://dati.normattiva.it) con `forge/scripts/prepare_legislation.py`.

### llama.cpp

```bash
llama-server -m legal-it-4b-Q4_K_M.gguf -c 8192 --jinja
```

Poi si invia il prompt sopra come unico messaggio utente all'endpoint `/v1/chat/completions`, con `"temperature": 0`.

### Ollama

```bash
ollama run hf.co/eullm/legal-it-4b:Q4_K_M
```

## Risultati

### Esami riservati (429 domande)

Le domande sono costruite automaticamente dal testo degli articoli: che cosa prevede un articolo, quale termine fissa (chiedendolo per numero di articolo o per argomento), e articoli che non esistono. Gli articoli degli esami sono esclusi da tutti i dati di addestramento. Tutti i modelli usano lo stesso retrieval descritto sopra.

| Modello | Corrette, su 429 | Senza giudice, su 219 |
|---|---|---|
| [legal-it-8b](https://huggingface.co/eullm/legal-it-8b) | 382 (89,0%) | 202 (92,2%) |
| **legal-it-4b** | **365 (85,1%)** | **200 (91,3%)** |
| Qwen3-4B-Instruct-2507, di partenza | 333 (77,6%) | 183 (83,6%) |

Il 4B vale quanto la versione dell'8B addestrata con il solo SFT (363 su 429).

Contro il modello di partenza, domanda per domanda (test esatto di McNemar):

- con il giudice, solo le risposte corrette: 51 domande giuste solo per legal-it-4b, 19 solo per il modello di partenza;
- contando giuste anche le risposte parziali: 46 a 12;
- senza giudice: 22 a 5.

Tutte e tre le differenze sono significative (p < 0,01).

### Come sono valutate le risposte

- **Con il giudice.** Qwen3-30B-A3B-Instruct-2507 confronta ogni risposta con il testo dell'articolo e la classifica come corretta, parziale o sbagliata. Su 40 risposte valutate alla cieca anche in modo indipendente, l'accordo è stato tra il 68% e l'85% a seconda del criterio. Per questo ogni confronto è riportato anche contando giuste le risposte parziali e senza giudice.
- **Senza giudice.** Le domande sui termini e sugli articoli inesistenti (219 su 429) si verificano in modo automatico: il termine indicato deve essere uno di quelli scritti nell'articolo, e su un articolo inesistente il modello deve dire che non c'è senza inventare termini.

### Quantizzazione

Il file Q4_K_M è stato valutato contro la versione bf16 su un esame di sviluppo di 472 domande, con gli stessi prompt e lo stesso retrieval (con il reranker Qwen3-Reranker-4B). Il Q4_K_M ha risposto tramite llama-server.

| | bf16 | Q4_K_M |
|---|---|---|
| Corrette, su 472 | 426 | 422 |
| Senza giudice, su 190 | 176 | 176 |

Domanda per domanda, 8 risposte sono giuste solo con il Q4 e 12 solo con il bf16 (p = 0,50), e senza giudice 3 a 3: la quantizzazione non costa nulla di misurabile.

## Addestramento

1. **Base:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache 2.0.
2. **SFT a libro aperto:** 18.972 esempi nel formato del prompt sopra.
   - L'85% sono domande e risposte scritte da Qwen3-30B-A3B-Instruct-2507 a partire da un singolo articolo, con l'indicazione di non aggiungere nulla che il testo non dica. Il prompt di addestramento contiene però i testi che il retrieval restituisce per quella domanda, così il modello impara a trovare quello giusto tra gli altri.
   - Il 15% sono domande su articoli inesistenti, con la nota nel prompt e una risposta che dice che l'articolo non c'è.
   - LoRA rank 32, learning rate 1e-4, 1 epoca, sequenze fino a 4.096 token, batch effettivo 16. LoRA unito ai pesi.
3. **GRPO con ricompense verificabili:** domande sui termini, chieste per articolo o per argomento, domande su articoli inesistenti e domande a cui il proprio articolo è stato tolto dai testi.
   - La ricompensa è calcolata da un programma, senza modelli: il termine giusto per le domande sui termini, l'astensione dove il testo non contiene la risposta.
   - LoRA rank 32, learning rate 1e-5, 250 passi, 8 risposte per prompt. LoRA unito ai pesi.
4. **Conversione:** GGUF F16 con llama.cpp, poi Q4_K_M con `llama-quantize`.

**Dati.** Solo testi normativi pubblici, da Normattiva: i codici civile, penale, di procedura civile e penale, del consumo e del processo amministrativo, la Costituzione, la legge n. 241/1990 e il d.P.R. n. 1199/1971, nel testo vigente scaricato da Normattiva nel 2026. Né l'SFT né il GRPO usano sentenze o dati personali.

**Calcolo.** L'addestramento è stato svolto sul supercomputer Leonardo di CINECA, nell'ambito dell'allocazione EuroHPC AI Factory EHPC-AIF-2026PG01-1147. Ringraziamo l'EuroHPC Joint Undertaking per l'accesso a Leonardo, ospitato da CINECA (Italia).

## Limiti

- **Copertura.** Conosce le norme che gli vengono date nel prompt. La qualità dipende dal retrieval e dalla raccolta normativa: norme abrogate o modificate dopo la raccolta (2026), leggi non incluse nella raccolta e giurisprudenza non sono coperte.
- **Errori.** Circa una risposta su sette sugli esami riservati è sbagliata o incompleta. Sui termini sbaglia meno, ma sbaglia.
- **Valutazione automatica.** I risultati vengono da un giudice automatico e da controlli automatici, non da una revisione di giuristi.
- **Lingua.** Addestrato e valutato solo in italiano.
- **Uso non previsto.** Non sostituisce un avvocato né la lettura del testo ufficiale. Non va usato per decisioni con conseguenze legali senza verifica umana.

## Licenza

Apache 2.0, come il modello di partenza Qwen3-4B-Instruct-2507. I testi normativi italiani non sono coperti da diritto d'autore (art. 5, legge n. 633/1941).

---

## English summary

**legal-it-4b** is a 4B model, fine-tuned from Qwen3-4B-Instruct-2507, that answers questions on Italian law **from the legal texts placed in its prompt** (retrieval-augmented, open book). The Q4_K_M GGUF is 2.5 GB.

- **Held-out results:** on 429 held-out questions it answers 85.1% correctly, against 77.6% for its base model with the same retrieval; the paired difference is significant under every criterion. [legal-it-8b](https://huggingface.co/eullm/legal-it-8b) reaches 89.0%.
- **Quantization:** Q4_K_M scores the same as bf16 (426 vs 422 of 472, p = 0.50).
- **Training:** open-book SFT on 18,972 pairs written from public legislation (Normattiva), then GRPO with programmatic rewards (deadlines, abstention on missing articles).
- **Intended use:** with the retrieval described above (BM25 + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B).
- **Not legal advice.**
- **Compute:** trained on Leonardo (CINECA) through the EuroHPC AI Factory allocation EHPC-AIF-2026PG01-1147.
