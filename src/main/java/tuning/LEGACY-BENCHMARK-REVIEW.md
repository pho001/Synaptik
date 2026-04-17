# Legacy Benchmark Review

Tento dokument slouží jako historická poznámka: co z původní benchmark/autotune vrstvy mělo smysl zachovat a co už dnes nemá být architektonickým vzorem.

Nejde o roadmapu návratu ke starému designu. Spíš o vysvětlení, proč dnešní `tuning` balík vypadá tak, jak vypadá.

## Historical Problem

Starší benchmark-first přístup typicky trpěl tím, že:

- benchmark měl vlastní kandidátový model
- runtime měl jiný execute model
- persistence ukládala něco jiného, než se pak opravdu spouštělo

Výsledek:

- složitá údržba
- drift mezi benchmarkem a runtime
- obtížná interpretace winners

## What Was Worth Keeping

I ze starší vrstvy ale mělo smysl zachovat některé věci.

### 1. Practical workload knowledge

Například:

- matmul scénáře
- conv2d scénáře
- transformer hot-path workloady

Tohle je cenné, protože workload know-how je drahé a nemá smysl ho ztrácet jen proto, že se mění orchestrace.

### 2. Some data factories and scenario helpers

Pokud dobře vystihují reálné tvary workloadů, mají hodnotu i po přepisu orchestrace.

### 3. Some measurement ideas

Například oddělení:

- compile
- prepare
- traced run
- steady-state

Tohle byl dobrý směr a dnešní tuning ho zachovává.

## What Became Legacy

### 1. Benchmark-Owned Candidate Universe

Starý model typu "benchmark kandidát" oddělený od `ExecutionProfile` už nemá být hlavní cesta.

Dnes je zdroj pravdy:

- `ExecutionProfile`

### 2. Monolithic Autotune Flow

Jedna obří třída, která:

- generuje kandidáty
- měří
- validuje
- ukládá
- rozhoduje strategii
- renderuje výsledky

je špatný design.

Dnešní tuning balík to rozděluje na:

- `candidate`
- `measure`
- `validate`
- `search`
- `store`
- `report`
- `session`

### 3. Benchmark-Specific Persistence

Persistence vázaná jen na starý benchmark runner nedávala smysl, protože:

- nebyla reuse-friendly
- míchala source of truth s explain data

## What Must Not Return

Tyto anti-patterny se nemají vracet pod jiným názvem:

- druhý skrytý execution model vedle `ExecutionProfile`
- benchmark-only knob universe
- ukládání reportu jako source of truth
- syntetické kandidátové modely, které runtime nikdy přímo nespustí

## What The New Architecture Replaced It With

Dnešní stav:

- benchmark měří explicitní `ExecutionProfile` kandidáty
- autotune searchuje explicitní `ExecutionProfile` kandidáty
- platform calibration mutuje explicitní `PlatformRuntimeProfile`
- persistence rozlišuje:
  - runtime defaults
  - best profile
  - history
  - explain artifacts

To je podstatně čistší než benchmark-first architektura.

## Keep / Freeze / Retire

### Keep

- workload know-how
- rozumné scenario builders
- užitečné measurement patterns

### Freeze

- historické compatibility fallbacky
- staré path layouty v `build/...`

### Retire

- starý benchmark-first candidate mindset
- staré dokumentační popisy, které prezentují benchmark jako hlavní architekturu runtime

## Practical Rule For New Work

Když dnes přidáváš novou tuning funkcionalitu, polož si otázku:

- jde to vyjádřit jako `ExecutionProfile` nebo `PlatformRuntimeProfile`?

Pokud ne, je vysoká šance, že znovu zavádíš starý problém.

## Why This Document Still Exists

Protože pomáhá vysvětlit:

- proč tuning odděluje platform defaults od workload winnerů
- proč je `ExecutionProfile` jediný execute source of truth
- proč reporty a history nejsou runtime artifacts

Tohle nejsou jen preferenční volby. Jsou to obranné mechanismy proti regresi architektury zpátky do benchmark-first guláše.
