# IR-Analyse
Analyse-Programm für IR-Spektroskopie

## Windows (PC / PowerShell)

### Setup

```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

### Windows -> .exe

```powershell
.\.venv\Scripts\Activate.ps1; pyinstaller --onefile --name IR-Analyse main.py
```

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
source .venv/bin/activate && python -m pip install --upgrade pyinstaller && pyinstaller --windowed --name "IR-Analyse" main.py
```

</details>
