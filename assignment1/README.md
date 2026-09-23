# 과제 1: Imitation learning

데이터 수집/추론과 서버 채점은 모두 **동일한 고정 트랙 1개**(시드 `22597174`)를 사용합니다.

**해야 할 일은 네 가지입니다.**

1. 아래 명령어로 **환경을 설치**합니다.
2. **[src/network.py](src/network.py)의 모델 구조를 개선**합니다.
3. **데이터 수집 → 모델 학습 → 모델 추론** 명령어를 실행합니다.
4. **제출 사이트에 `network.py`와 `.pth`를 제출**합니다.

제출 장소: (추후 공개 예정)
임시 제출 장소(학교 내부망에서 접속 가능): `http://166.104.224.209:8000/`

## 1. 환경 설치 — 최초 한 번

저장소의 `assignment1/` 폴더에서 **본인 운영체제 명령어만** 실행하세요. Python 3.12, 64비트 기준이며 GPU는 자동 감지합니다.

<details>
<summary>Windows / PowerShell</summary>

Python 3.12가 없다면 설치 후 PowerShell을 다시 여세요.

```powershell
winget install --exact --id Python.Python.3.12 --source winget
```

```powershell
py -3.12 -m venv .venv
$env:Path = "$PWD\.venv\Scripts;$env:Path"
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe scripts/setup.py configs/config.yaml
```

Windows ARM64는 Visual Studio 2022 Build Tools의 C++ 데스크톱 개발, MSVC ARM64/ARM64EC 빌드 도구와 Windows 11 SDK가 추가로 필요합니다.

</details>

<details>
<summary>macOS / 터미널 (Homebrew 설치 필요)</summary>

```bash
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python scripts/setup.py configs/config.yaml
```

</details>

<details>
<summary>Ubuntu 24.04 / 터미널</summary>

```bash
sudo apt update
sudo apt install -y python3.12-venv python3.12-dev build-essential libgl1 libsdl2-2.0-0
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python scripts/setup.py configs/config.yaml
```

</details>

이후 명령은 모두 **assignment1/ 폴더에서** 실행합니다.

## 2. 모델 구조 개선

[src/network.py](src/network.py)의 **구현 내용**(`self.conv`, `self.fc`)을 수정하세요.
입력 `(B, 96, 96, 3)`, 출력 `(B, 9)`와 제공된 행동 클래스 순서는 유지합니다.

## 3. 데이터 수집 → 모델 학습 → 모델 추론

<details>
<summary>Ubuntu / macOS</summary>

```bash
# 1) 데이터 수집
.venv/bin/python scripts/collect.py configs/config.yaml

# 2) 모델 학습
.venv/bin/python scripts/train.py configs/config.yaml

# 3) 모델 추론 — 주행 확인
.venv/bin/python scripts/infer.py configs/config.yaml
```

</details>

<details>
<summary>Windows / PowerShell</summary>

```powershell
# 1) 데이터 수집
.\.venv\Scripts\python.exe scripts/collect.py configs/config.yaml

# 2) 모델 학습
.\.venv\Scripts\python.exe scripts/train.py configs/config.yaml

# 3) 모델 추론 — 주행 확인
.\.venv\Scripts\python.exe scripts/infer.py configs/config.yaml
```

</details>


## 4. 과제 제출

**제출 사이트**에 실명으로 가입/로그인합니다. 실명과 비밀번호로 바로 가입할 수 있습니다.

- **모델 구조:** 학습할 때 자동 저장된 `models/날짜.network.py`
- **모델 가중치:** 같은 이름의 `models/날짜.pth`

구조 파일은 `network.py`의 저장 당시 사본입니다. **같은 학습에서 생성된 두 파일을 함께 제출**하세요.
학습 후 수정한 현재 `src/network.py`를 이전 `.pth`와 섞어 제출하면 평가에 실패합니다.
제출 후 사이트에서 대기 상태, 주행 영상, 평가 지표와 순위를 확인할 수 있습니다.
