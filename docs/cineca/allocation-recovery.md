# Rientrare a lavoro — cosa muove davvero i nodi

Scritto il 22 settembre 2026, dopo 26 ore senza che nessun job partisse su
`boost_usr_prod` e con il 20 % delle node-hour consumate a fronte del 34 %
del calendario.

Questo documento distingue le leve reali da quelle che sembrano leve. La
distinzione conta perché il tempo speso su una leva finta è tempo in cui
l'allocazione continua a scorrere.

## La ricetta, come funziona oggi (aggiornata al 25 settembre)

Quello che segue il 22 settembre era un'indagine; questa sezione è il
risultato, raccolto in un posto solo perché fino a oggi stava sparso fra i
commenti dei lanciatori e le conversazioni. Il consumo giornaliero, misurato
con lo stesso metodo (`sacct`, billing/32):

| giorno | node-hour |
|---|---|
| 23/09 | 25,0 |
| 24/09 | 38,9 |
| 25/09 | 36,7 alle 16:25, proiezione ~50 |

contro una media storica di 12,3 con giornate a zero.

### 1. Una fetta di nodo, non un nodo: 3 GPU, 24 core, 340 GB

Due probe inviati nello stesso secondo, identici tranne `--gres`: `probe-g3`
parte in **87 s**, `probe-g4` resta pending (vedi l'intestazione di
`forge/scripts/leonardo/sbatch_phase2_split.slurm`). Uno snapshot `sinfo` del
22/09 contava **1 nodo idle contro 364 mixed**: un nodo con una GPU libera è
normale, un nodo intero libero no.

Core e memoria fanno parte della stessa richiesta. Un nodo Booster ha 32 core:
`--cpus-per-task=32` rende il job esclusivo qualunque sia il numero di GPU, e
450 GB su ~494 lasciano troppo poco a chiunque altro. Per questo **24 core e
340 GB**. Il numero di GPU dichiarato allo `--expect-gpus` del pre-flight deve
cambiare insieme a `--gres`, o ogni anello muore nel pre-flight.

### 2. Anelli da due ore

Probe da 30 minuti e da 1 ora collocati in 50 s; 2 ore collocato; 3, 4 e 6
ore **mai** collocati dopo oltre 22 ore. La soglia sta fra 2 e 3 ore. Con ~14
minuti di avvio (22 per l'8B a freddo) un anello da 2 ore rende ~80 % delle
node-hour che spende (tabella più sotto). Gli anelli finiscono in TIMEOUT per
progetto: il successore riprende dall'ultimo checkpoint.

### 3. `save_steps` sotto la lunghezza dell'anello

A ~380 step/ora un anello da 2 ore fa ~670 step. `save_steps` deve stare ben
sotto (100 per r32, 300 per lo split), altrimenti un anello finisce senza aver
salvato e il successivo riparte dallo stesso punto: tempo speso, zero
avanzamento, nessun errore visibile.

### 4. Catene concorrenti, una per esperimento

Ogni catena `afterany` ha **un solo job eleggibile**, il capo; gli altri
anelli aspettano in `Dependency` e non contano nemmeno per
`bf_max_job_user_part`. Tre catene indipendenti (split, r32, 8B) sono tre job
che corrono in parallelo. Una catena va allungata quando le restano meno di
~24 ore di anelli in coda.

### 5. Condizione indispensabile: la ripresa deve continuare i DATI

**Senza questa, tutto quello sopra peggiora il modello invece di
addestrarlo.** Fino alla PR #529 una ripresa ricaricava pesi, ottimizzatore e
step ma ripartiva dall'inizio dei dati: con anelli da 2 ore ogni anello
riaddestrava gli stessi ~650 step di corpus. Lo split lo ha misurato — PPL sul
CdS 5,11 allo step 12.600, 5,29 al 18.200, 5,92 al 22.000. Dopo la correzione
il log di ogni anello deve mostrare
`[resume] data: epoch 0, skipping the … batches of it already trained on`.
Qualunque nuovo script di training che lavori ad anelli deve avere la stessa
garanzia prima di essere messo in catena.

### 6. Tutto il lavoro CPU sulla seriale

Export GGUF, curva di transfer, impacchettamento dello stadio 3, prove sui
modelli: `lrd_all_serial`, che parte subito e non tocca le GPU. Il lavoro che
dipende da un altro job va accodato con `--dependency`, non lanciato a mano
quando "sembra finito".

### Correzione alla lettura della priorità

La tabella qui sotto confronta la nostra priorità (141.963) con il **massimo**
fra i pending, ed è corretto che siamo due ordini di grandezza sotto. Ma
rispetto alla **mediana** (138.373 su 3.688 job pending, dallo snapshot in
`docs/paper/measurements/priority_20260922.json`) eravamo **sopra**. Il
problema non era la priorità: era la forma della richiesta (punti 1 e 2).

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

## Quanto costa una finestra: le misure del 22 settembre

Tre numeri misurati la sera del 22, che insieme decidono la lunghezza di un
anello. Nessuno dei tre era noto prima, e due contraddicono le stime su cui
avevamo ragionato per mezza giornata.

**Un job corto viene collocato in cinquanta secondi.** Un probe identico alle
catene — un nodo, 4 A100, 32 core, 450 GB, QOS `normal` — con `--time=30:00`
ha preso un nodo in 50 s (submit 19:16:13, start 19:17:03). Lo stesso probe a
`--time=01:00:00` ci ha messo gli stessi 50 s. A 2, 3, 4 e 6 ore era ancora
pending mezz'ora dopo, contro le 28 ore delle teste da 4 ore già in coda.

Il cluster non è pieno: le finestre da un'ora ci sono sempre. Quello che non
esiste è un buco da quattro ore. Non stiamo aspettando risorse, stiamo
chiedendo una forma che non c'è.

Il probe si fattura sul tempo reale (5 s) e si colloca sul walltime richiesto,
per cui l'intera scala è costata meno di un minuto di nodo.

**Lo startup di un anello è di ~14 minuti**, da `eullm-p2b-split-57893877`:
3 minuti di pre-flight (import di transformers da Lustre) e 11 di caricamento
del teacher da 61 GB in BF16. Il riavvolgimento del dataloader alla ripresa da
`checkpoint-8400` è risultato trascurabile — una decina di secondi fra
"modelli caricati" e primo step — quindi quel pedaggio non lo paghiamo. Vale
la pena riguardarlo a step molto più alti, dove un eventuale salto
crescerebbe in proporzione.

**Il throughput è di ~379 step/ora**, misurato su otto intervalli consecutivi
di `logging_steps: 20` (~190 s ciascuno), non i 333 stimati.

### La conseguenza, che è il motivo per cui le misure servivano

| anello | training netto | step | salvati a `save_steps: 300` | utile |
|---|---|---|---|---|
| 1 h | 46 min | 291 | **nessuno** | 0 % |
| 1 h, `save_steps: 100` | 46 min | 291 | 200 | 53 % |
| 2 h | 106 min | 670 | 600 | 79 % |
| 3 h | 166 min | 1.048 | 900 | 79 % |
| 24 h | 23h46 | 9.010 | 9.000 | 99 % |

Un anello da un'ora con la configurazione attuale **produce zero**: 291 step
contro un primo salvataggio a 300. Girerebbe, morirebbe in TIMEOUT e il
successore ripartirebbe dallo stesso punto. Abbassare il walltime senza
toccare `save_steps` sarebbe stato un peggioramento invisibile per giorni.

E anche con `save_steps` corretto, un'ora rende ~12,5 node-ora utili al
giorno, contro una media storica di 12,3: un pareggio, non un guadagno.

**Gli anelli corti non sono un moltiplicatore, sono un pavimento.** Il che
serve comunque, perché quel 12,3 è la media di giornate a 36 e di sette
giornate a zero — `saldo -r` le elenca. Un anello corto non alza il tetto,
toglie gli zeri. Per questo la struttura giusta somma le due cose invece di
sceglierne una: le catene da 24 ore restano in coda per le finestre grandi al
99 % di efficienza, una catena ad anelli corti bruca in continuo, e i due si
sommano fino alla quota mensile.

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
