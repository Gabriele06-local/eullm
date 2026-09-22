# Rientrare a lavoro — cosa muove davvero i nodi

Scritto il 22 settembre 2026, dopo 26 ore senza che nessun job partisse su
`boost_usr_prod` e con il 20 % delle node-hour consumate a fronte del 34 %
del calendario.

Questo documento distingue le leve reali da quelle che sembrano leve. La
distinzione conta perché il tempo speso su una leva finta è tempo in cui
l'allocazione continua a scorrere.

## Il quadro, in numeri misurati

Dagli snapshot in `docs/paper/measurements/`, che esistono proprio perché
`sprio`, `sshare` e `sinfo` non conservano storia:

| | |
|---|---|
| Priorità dei nostri job | 141.963 |
| — di cui QOS | 120.000 |
| — di cui fairshare | 18.783 |
| — di cui age | 3.175 |
| — di cui **jobsize** | **6** |
| Priorità massima fra i pending della partizione | 60.259.060 |
| Fairshare | 0,751 (usage 0,000086 su 0,000208 di quota) |
| Avvii su boost negli ultimi dieci giorni | 16/09, 18/09, 21/09 |

I pesi dello scheduler, da `scontrol show config`:

```
PriorityWeightJobSize   = 10000000
PriorityWeightQOS       =   300000
PriorityWeightFairShare =    25000
PriorityWeightAge       =    20000
PriorityMaxAge          = 7-00:00:00
```

Un job da un nodo contribuisce 6 su dieci milioni. Il fairshare, con un uso
pari a un terzo della quota, non ci sta penalizzando: se fosse quello, la
soluzione sarebbe consumare meno, ed è l'opposto del problema.

**Un'avvertenza onesta sulla lettura.** La priorità massima osservata nella
partizione — 60.259.060 — è superiore alla somma di tutti i pesi elencati
sopra, quindi qualcosa contribuisce oltre a JobSize/QOS/FairShare/Age
(plausibilmente `PriorityWeightTRES` o un `PrioritySiteFactor`). Il fatto
resta — siamo dietro di due ordini di grandezza — ma l'attribuzione causale
al solo `PriorityWeightJobSize` è un'ipotesi, non una misura. È il motivo per
cui la mail al supporto la formula come domanda.

## La leva che conta: catene indipendenti concorrenti

Una catena `afterany` ha **un solo job eleggibile alla volta**: il capo. Gli
altri anelli sono PENDING con reason `Dependency` e non concorrono per nulla.
Tre catene significano tre job eleggibili, non ventidue.

Con la cadenza misurata — un nodo ogni due o tre giorni per catena, ≈ 4,8 h
di compute al giorno per catena — e 41 giorni residui:

| Catene | Node-hour previste | % dell'allocazione |
|---|---|---|
| 1 | ~197 | 36 % |
| 2 | ~394 | 52 % |
| 3 | ~590 | 68 % |
| 5 | ~985 | 99 % |

Il numero di catene è l'unica variabile con effetto lineare. Non il walltime,
non le GPU per job, non la QOS — quelle spostano di poco.

## Correzione: non serve cancellare nulla

Avevo indicato `bf_max_job_user_part=20` come vincolo, con 22 job in coda, e
suggerito di tagliare un anello. **Non fatelo: il vincolo non ci tocca.**

Quel parametro limita quanti job per utente e partizione il *backfill* esamina
per ciclo, e il backfill scarta i job non eleggibili — quelli con dipendenza
non soddisfatta — prima di contarli. Dei nostri 22 job in coda, 19 sono anelli
in attesa di dipendenza: invisibili al contatore. I job effettivamente
esaminati sono 3, i capi delle tre catene.

Conseguenza pratica, ed è la parte utile: **aggiungere catene non consuma quel
budget**. Possiamo arrivare a cinque o sei catene senza avvicinarci al limite,
perché ogni catena aggiunge un solo job eleggibile.

## Cosa fare, in ordine

1. **Portare le catene da 3 a 5.** È la leva lineare. La quarta è l'arm
   Consiglio di Stato descritto sotto; per la quinta l'ipotesi più economica è
   un'ablazione sul rank LoRA dell'arm 4B (128 → 32), che il report vuole
   comunque e che è un `student_lora_rank` diverso nello stesso YAML, con
   `output_dir` e `EULLM_RUN_DIR` propri.
2. **Tenere piena la partizione seriale.** `lrd_all_serial` parte subito e non
   tocca l'allocazione GPU. La curva di transfer, gli export GGUF e le misure
   di perplessità stanno tutti lì: è lavoro che avanza mentre boost tace.
3. **Inviare la richiesta al supporto CINECA.** La reservation è lo strumento
   che possono concedere davvero; la proroga di calendario è il ripiego che
   converte con certezza in compute. Il testo della richiesta non sta in
   questo repository — è corrispondenza con l'ente che ospita l'allocazione, e
   contiene nome utente e referenti del progetto.
4. **Da valutare, non da assumere: `boost_qos_dbg` come riempitivo.** Priorità
   80 contro 40, ma 30 minuti e 2 job. Con `save_steps: 300` e il caricamento
   del teacher da Lustre, buona parte della finestra se ne va prima del primo
   checkpoint, e un anello che non raggiunge un salvataggio è tempo buttato.
   Va misurato su un singolo job — quanti step in 30 minuti a freddo — prima
   di metterci una catena. Se il primo checkpoint non rientra, l'idea muore
   lì.

## L'arm Consiglio di Stato

Le sentenze del Consiglio di Stato che stiamo scaricando servono a due cose
diverse, e **non possono servirle entrambe sullo stesso anno**.

### Il vincolo, che viene prima di tutto il resto

`docs/corpus-acquisition.md` fissa una partizione temporale permanente:

* **2025 e 2026 — solo valutazione.** Non entrano in nessun training, mai.
* **2017-2024 — disponibili per il training.**

Il 23 % di miglioramento misurato allo step 28.000 vale perché è misurato su
un tribunale che il modello non ha mai letto. Addestrare sul 2025 lo
cancellerebbe, e cancellerebbe retroattivamente anche tutti i punti della
curva di transfer già misurati, perché non sarebbero più confrontabili con
quelli successivi. Non è una precauzione: è l'unico numero pulito che abbiamo.

Quindi sì, alleniamo anche sul Consiglio di Stato — **sul 2017-2024**.

### Perché è anche la quarta catena

Non è solo un modello in più. È un arm indipendente che gira su un nodo
proprio, con un suo `EULLM_RUN_DIR`, e che quindi aggiunge un job eleggibile
alla coda. Risolve il problema di questo documento e produce un deliverable
nello stesso movimento.

La variabile sotto test è **la dimensione e la composizione del corpus**:
teacher, student, adapter di fase 1, rank, obiettivo, schedule e seed restano
identici all'arm split. Cassazione da sola contro Cassazione + Consiglio di
Stato 2017-2024, a step appaiati, sullo stesso held-out. È una domanda a cui
il report vuole rispondere e che finora non potevamo porre.

### Cosa manca prima di lanciarla

1. **Il download**, che è la parte lunga: l'indice OpenGA copre ~70.000
   sentenze per il 2017-2024 e il testo pieno si prende una richiesta per
   sentenza, con il ritardo di cortesia. Va fatto in modalità non presidiata e
   riavviabile — è l'unica voce di backlog che blocca il resto.
2. **La pipeline di processing**, che esiste già e non va toccata:
   anonimizzazione → `sweep_structured_pii.py` come gate → dedup →
   `format_pretraining.py --group-by sentence_id`.
3. **Il filtro sull'anno**, che è la sola riga nuova e la sola che, se
   sbagliata, non si vede: il formatter deve rifiutarsi di includere record
   del 2025-2026, e va verificato contando i record per anno nel `train.jsonl`
   prodotto, non fidandosi del flag.

La config e il launcher dell'arm sono in
`forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_cds.yaml` e
`forge/scripts/leonardo/sbatch_phase2_cds.slurm`. Entrambi falliscono in
partenza se il corpus combinato non c'è, invece di allenarsi in silenzio sul
corpus vecchio.
