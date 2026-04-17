# Tuning Persistence

Persistence v tuning vrstvě neslouží jen k "uložení nějakého JSONu". Musí přesně rozlišovat, co je:

- execute source of truth
- tuning prior
- explain artifact

Pokud se tyhle vrstvy smíchají, začne drift mezi tím, co se benchmarkovalo, co se spouští a co se ukládá jako "vítěz".

## Reading Guide

Tento dokument popisuje:

- jaké perzistentní artefakty dnes existují
- kde mají žít
- jaké mají lifecycle
- které z nich se používají pro execute a které jen pro explain/search priors

## Artifact Types

### 1. Built-In Defaults

Žijí v kódu.

Nejsou perzistentní tuning artifact.

### 2. `PlatformRuntimeProfile`

Výsledek platform calibration.

Je to machine-specific runtime default artifact.

Používá se jako skutečný vstup do `ExecutionProfileAssembler`.

### 3. Best `ExecutionProfile`

Výsledek graph autotune pro konkrétní workload/hardware kontext.

Je to workload-specific runnable winner.

### 4. Tuning History

Append-only evidence o kandidátech:

- valid/invalid
- median/mean
- score
- summary

Používá se jako prior pro history-aware search.

### 5. Explain Artifacts

Sem patří:

- text/json benchmark reports
- text/json autotune reports
- text/json platform calibration reports

Nepoužívají se jako runtime source of truth.

## Preferred Layout Today

Preferovaný layout je platform-versioned storage pod:

- `profiles/platform/<platform-id>/...`

Konkrétní pattern používaný v [Main.java](../synaptik/app/Main.java):

### Platform calibration

- `profiles/platform/<platform-id>/calibration/<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.txt`

### Graph autotune

Pro workload `abc`:

- `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json`
- `profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl`

## Compatibility Fallbacks

Repo stále umí číst starší fallback layouty pod `build/...`:

- `build/platform-calibration/...`
- `build/tuning/best-profiles/...`
- `build/tuning/history/...`

Ale tohle už není preferovaný dlouhodobý layout.

Dokumentace to musí říkat explicitně:

- `build/...` je compatibility / temporary output space
- `profiles/platform/...` je preferované místo pro versioned persisted tuning state

## Platform Runtime Profile Persistence

Hlavní typy:

- [PlatformRuntimeProfile.java](../config/profile/PlatformRuntimeProfile.java)
- [PlatformRuntimeProfileIO.java](../config/profile/PlatformRuntimeProfileIO.java)
- [PlatformRuntimeProfileStore.java](./store/PlatformRuntimeProfileStore.java)
- [JsonFilePlatformRuntimeProfileStore.java](./store/JsonFilePlatformRuntimeProfileStore.java)

Ukládaný obsah:

- metadata
- matmul family
- fused family
- elementwise dispatch family
- reduction family
- scheduler family
- materialization family
- numerics family

Význam:

- můžeš recalibrovat jen část rodin a zbytek ponechat
- můžeš stejný runtime profil použít napříč více benchmark/autotune workloady

## Best Profile Persistence

Hlavní typy:

- [BestProfileRecord.java](./store/BestProfileRecord.java)
- [BestProfileStore.java](./store/BestProfileStore.java)
- [JsonFileBestProfileStore.java](./store/JsonFileBestProfileStore.java)

JSON dnes ukládá:

- `score`
- `updatedAt`
- `hardwareKey`
- `workloadKey`
- embedded `ExecutionProfile`

To je důležitá realita:

- best profile record není jen holý `ExecutionProfile`
- obsahuje i identitu kontextu, pro který platí

## Tuning History Persistence

Hlavní typy:

- [TuningHistoryEntry.java](./store/TuningHistoryEntry.java)
- [TuningHistoryStore.java](./store/TuningHistoryStore.java)
- [JsonFileTuningHistoryStore.java](./store/JsonFileTuningHistoryStore.java)

Formát:

- JSON Lines
- jeden candidate observation per řádek

Dnešní fields:

- `fingerprint`
- `candidateName`
- `valid`
- `medianMs`
- `meanMs`
- `score`
- `failureReason`
- `summary`
- `timestamp`
- `hardwareKey`
- `workloadKey`

To dává smysl pro:

- history-aware ordering
- prune invalid candidates
- zachování audit trailu bez přepisování minulosti

## Explain Artifact Persistence

Platform calibration report persistence:

- [PlatformCalibrationSaveHelper.java](./store/PlatformCalibrationSaveHelper.java)
- [JsonFilePlatformCalibrationResultStore.java](./store/JsonFilePlatformCalibrationResultStore.java)

Platí:

- JSON i text report jsou explain artifacts
- runtime source of truth je pořád samotný `PlatformRuntimeProfile`

## Fingerprints

### Hardware Fingerprint

Typ:

- [HardwareFingerprint.java](./store/HardwareFingerprint.java)

Použití:

- platform runtime profile reuse
- best profile reuse
- tuning history filtering

### Workload Fingerprint

Typ:

- [WorkloadFingerprint.java](./store/WorkloadFingerprint.java)

Použití:

- rozlišení workload-specific best profile a history

Best profile resolver je schválně přísný:

- hardware key musí sedět
- workload key musí sedět

Viz:

- [FileBestProfileResolver.java](./store/FileBestProfileResolver.java)

## Invalidation Rules

Perzistentní tuning artifact není věčný.

### Platform Runtime Profile invaliduj, když:

- se změnil hardware fingerprint
- se změnila semantika runtime knobů
- se změnil schema formát
- se změnil framework/runtime behavior tak, že staré winners nedávají smysl

### Best Profile invaliduj, když:

- nesedí hardware fingerprint
- nesedí workload fingerprint
- změnil se schema nebo význam polí v `ExecutionProfile`

### History invaliduj nebo ignoruj, když:

- workload/hardware keys nesedí
- candidate fingerprints už neodpovídají dnešnímu candidate space

## Source Of Truth Rules

### Co je execute source of truth

- `PlatformRuntimeProfile`
- `ExecutionProfile`

### Co není execute source of truth

- tuning history
- benchmark report
- calibration report
- tuning summary text

Tohle pravidlo nesmí dokumentace rozmělňovat.

## Example: Save Platform Calibration

```java
PlatformCalibrationSaveHelper.saveAll(
        result,
        layout.profilePath(),
        layout.jsonReportPath(),
        layout.textReportPath()
);
```

Tím se uloží:

- finální runtime profile
- JSON report
- text report

## Example: Save Best Profile

```java
bestProfileStore.save(path, new BestProfileRecord(
        hardware,
        workload,
        bestProfile,
        score,
        java.time.OffsetDateTime.now()
));
```

## Example: Append History

```java
historyStore.append(path, new TuningHistoryEntry(
        fingerprint,
        candidateName,
        valid,
        medianMs,
        meanMs,
        score,
        failureReason,
        summary,
        java.time.OffsetDateTime.now(),
        hardware,
        workload
));
```

## Practical Recommendations

- drž `profiles/platform/...` v repozitářem známém místě
- `build/...` používej jen jako fallback nebo scratch prostor
- reporty a source-of-truth profily neukládej do stejného souboru
- history nech append-only
- best profile přepisuj jen při skutečném lepším winneru

## Common Mistakes

- ukládat report jako náhradu za profile artifact
- používat history JSONL jako runtime config
- ignorovat hardware/workload fingerprint při reloadu
- míchat platform defaults a graph winner do jednoho souboru bez identity

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- reporting: [REPORTING.md](./REPORTING.md)
- search: [SEARCH.md](./SEARCH.md)
