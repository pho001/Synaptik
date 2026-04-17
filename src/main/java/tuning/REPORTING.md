# Tuning Reporting

Reporting vrstva dělá dvě věci:

- převádí výsledky do lidsky čitelné podoby
- převádí výsledky do strojově čitelných explain artifactů

Nedělá:

- execution
- candidate search
- persistence rozhodnutí

## Reading Guide

Použij tento dokument, pokud chceš pochopit:

- co dnes benchmark/tuning/calibration report opravdu obsahují
- odkud se berou compile/prepare/run čísla
- kde se v text reportech objevují hot steps
- co patří do progress eventů a co do finálních reportů

## Benchmark Reporting

Hlavní typy:

- [BenchmarkCandidateReport.java](./report/BenchmarkCandidateReport.java)
- [BenchmarkReport.java](./report/BenchmarkReport.java)
- [BenchmarkSuiteReport.java](./report/BenchmarkSuiteReport.java)
- [TextBenchmarkReportRenderer.java](./report/TextBenchmarkReportRenderer.java)

Benchmark report dnes odpovídá na:

- který kandidát vyhrál
- kolik kandidátů uspělo/neuspělo
- jaké byly steady-state časy
- jak si kandidáti stojí vůči baseline
- jaké byly hot runtime kroky ve traced běhu

### Co `TextBenchmarkReportRenderer` dnes opravdu ukazuje

Souhrnnou tabulku s poli:

- `name`
- `status`
- `compileMs`
- `prepareMs`
- `traceMs`
- `medianMs`
- `p90Ms`
- `vsBaseline`

A potom pro každého kandidáta detail:

- validation status
- optimizer stage order
- compile/prepare/traced run čas
- step count
- `parallelUsed`
- `vectorUsed`
- steady-state mean/median/p90
- `speedupVsBaseline`
- top hot steps
- full step dump s trace metadata

To znamená, že benchmark reporting není jen "jméno a median". Je to užitečný výkonový diagnostický artifact.

## Autotune Reporting

Hlavní typy:

- [TuningResult.java](./session/TuningResult.java)
- [TuningSummary.java](./report/TuningSummary.java)
- [TextTuningResultRenderer.java](./report/TextTuningResultRenderer.java)

Autotune reporting odpovídá na:

- který `ExecutionProfile` vyhrál
- jaká search strategie byla použita
- kolik kandidátů bylo vybráno/evaluováno
- kolik jich prošlo validation
- kolik history entries se zapsalo

### Co dnes ukazuje `TextTuningResultRenderer`

- `bestProfile`
- `persisted`
- `summary`
- `strategy`
- `selected`
- `evaluated`
- `valid`
- `finalists`
- `historyEntriesWritten`
- `bestMedianMs`

a potom tabulku finalistů:

- name
- median
- mean
- validation status

## Platform Calibration Reporting

Hlavní typy:

- [PlatformCalibrationResult.java](./session/PlatformCalibrationResult.java)
- [PlatformCalibrationStepResult.java](./session/PlatformCalibrationStepResult.java)
- [PlatformCalibrationCandidateSummary.java](./session/PlatformCalibrationCandidateSummary.java)
- [PlatformCalibrationScore.java](./session/PlatformCalibrationScore.java)
- [TextPlatformCalibrationResultRenderer.java](./report/TextPlatformCalibrationResultRenderer.java)
- [JsonPlatformCalibrationResultRenderer.java](./report/JsonPlatformCalibrationResultRenderer.java)

Platform calibration report odpovídá na:

- jaký byl seed runtime profile
- které family kroky proběhly
- jaký kandidát v každém kroku vyhrál
- jaká score metrika byla použita
- jaký finální runtime profile vznikl

### Co dnes ukazuje `TextPlatformCalibrationResultRenderer`

- `platformId`
- `createdAt`
- `persisted`
- `outputProfilePath`
- `profileName`
- `dataType`
- `mode`
- `seedRuntimeProfile`
- `finalRuntimeProfile`
- hardware summary
- tabulku kroků:
  - `name`
  - `family`
  - `seedRuntime`
  - `selectedExec`
  - `score`
  - `metric`

## Trace-Derived Reporting

Reporting sám trace negeneruje.

Trace přichází z execution vrstvy přes `MeasurementResult.trace()`.

Typicky obsahuje:

- compile trace
- prepare trace
- run trace
- step traces

Benchmark renderer pak z něj umí vytáhnout:

- compile/prepare duration
- traced cold run duration
- hot steps
- layout/dispatch/reduction/matmul/fused metadata na každém stepu

## Why Reporting Is Separate

Reporting nesmí rozhodovat:

- jestli je kandidát validní
- jak se měří
- co se uloží do persistence

Má jen převést již existující DTO do výstupu.

Tím je zajištěné, že:

- text renderer nemění semantics výsledků
- JSON renderer není execute source of truth

## Progress Reporting

Vedle finálních reportů existují i live progress eventy.

### Autotune progress

Typy:

- [AutotuneProgressEvent.java](./session/AutotuneProgressEvent.java)
- [AutotuneProgressPhase.java](./session/AutotuneProgressPhase.java)

Current phases:

- `STARTED`
- `SEARCH_BATCH`
- `CANDIDATE_VALIDATING`
- `CANDIDATE_INVALID`
- `CANDIDATE_MEASURING`
- `CANDIDATE_MEASURED`
- `CANDIDATE_FAILED`
- `ROUND_COMPLETED`
- `COMPLETED`

### Platform calibration progress

Typy:

- [PlatformCalibrationProgressEvent.java](./session/PlatformCalibrationProgressEvent.java)
- [PlatformCalibrationProgressPhase.java](./session/PlatformCalibrationProgressPhase.java)

Current phases:

- `STARTED`
- `FAMILY_STARTED`
- `WORKLOAD_STARTED`
- `CANDIDATE_VALIDATING`
- `CANDIDATE_INVALID`
- `CANDIDATE_MEASURING`
- `CANDIDATE_MEASURED`
- `CANDIDATE_FAILED`
- `CANDIDATE_SCORED`
- `FAMILY_COMPLETED`
- `COMPLETED`
- `FAILED`

Tyto eventy nejsou náhrada finálního reportu. Jsou určeny pro:

- dlouhý běh v terminálu
- CI log visibility
- debugging stalls a candidate explosions

## JSON Expectations

JSON renderery mají být:

- machine-readable explain artifacts
- vhodné pro diffing nebo archivaci

Nemají být:

- execute source of truth
- jediný persistence artifact

Source of truth zůstává:

- `PlatformRuntimeProfile` pro platform defaults
- `ExecutionProfile` pro graph winner

## Example: Benchmark Report Interpretation

Pokud benchmark report ukáže:

- `bestMedianMs` lepší než baseline
- ale `traceMs` horší

může to znamenat:

- vyšší cold/traced overhead
- ale lepší steady-state

Proto se vyplatí sledovat:

- compile/prepare/traced run
- steady-state median
- hot steps

ne jen jedno číslo.

## Example: Calibration Report Interpretation

Pokud v platform calibration reportu uvidíš:

- dobré matmul score
- ale pozdější fused family zhoršení

je to očekávané v sekvenčním family flow:

- každý další krok už startuje z předchozího vítěze
- report zachovává audit trail, který krok co změnil

## Common Mistakes

- používat text report jako jediný perzistentní artifact
- číst speedup bez kontextu validation statusu
- zaměnit traced cold run za steady-state median
- ignorovat hot-step dump při výkonové regresi

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- search: [SEARCH.md](./SEARCH.md)
