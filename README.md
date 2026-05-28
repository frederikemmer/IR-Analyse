# IR-Analyse
Analyse-Programm für IR-Spektroskopie

## Windows (PC / PowerShell)

### Setup

```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

### Windows -> .exe

```powershell
.\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pyinstaller; python -m PyInstaller --onefile --name IR-Analyse main.py
```

### Modelle neu trainieren

```powershell
.\.venv\Scripts\Activate.ps1; $env:IR_ANALYSE_TEST_MODE='1'; python -c "import main as app; report = app.ML.retrain_all_models(archive_dir='models/archiv'); print(report['version'])"
```

Aktive Modelle werden in `models/*.model` aktualisiert. Versionierte Backups landen zusätzlich in `models/` und `models/archiv/`. Ein Trainingsreport wird nach `models/training_report.json` geschrieben.

---

<br>

<details>
<summary>macOS / Linux anzeigen</summary>

## macOS / Linux

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
```

### macOS -> .app

```bash
source .venv/bin/activate && python -m pip install --upgrade pyinstaller && python -m PyInstaller --windowed --name "IR-Analyse" main.py
```

### Modelle neu trainieren

```bash
source .venv/bin/activate && IR_ANALYSE_TEST_MODE=1 python -c "import main as app; report = app.ML.retrain_all_models(archive_dir='models/archiv'); print(report['version'])"
```

Aktive Modelle werden in `models/*.model` aktualisiert. Versionierte Backups landen zusätzlich in `models/` und `models/archiv/`. Ein Trainingsreport wird nach `models/training_report.json` geschrieben.

</details>
