# OCR Captcha API

API HTTP que recebe a imagem de um captcha e devolve o texto. Usa uma CRNN pequena
(rede convolucional + LSTM, ~1 milhão de parâmetros, ~4 MB) treinada neste repositório
para o estilo dos captchas de `captchas/rotulados`: 200×68 em tons de cinza, 4 a 6 letras
minúsculas e dígitos, linhas onduladas e ruído de pontos.

O modelo treinado fica em `models/crnn.pt` e já vem no repositório.

## Requisitos

- Python 3.10 ou superior
- Roda bem na CPU: cada captcha leva poucos milissegundos.

## 1. Instalar

```bash
git clone https://github.com/raulfranck/ocr-captcha.git
cd ocr-captcha
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Só CPU (download bem menor). Pule esta linha se tiver GPU NVIDIA.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## 2. Configurar (opcional)

```bash
cp .env.example .env
```

| Variável          | Padrão              | O que faz                                                    |
|-------------------|---------------------|--------------------------------------------------------------|
| `MODEL_PATH`      | `./models/crnn.pt`  | Modelo treinado por `scripts/train_crnn.py`                  |
| `DEVICE`          | `auto`              | `cpu`, `cuda`, `mps` ou `auto`                               |
| `MAX_IMAGE_BYTES` | `2000000`           | Tamanho máximo do upload                                     |
| `API_KEY`         | vazio               | Se definido, exige o header `X-API-Key` em todas as chamadas |

## 3. Rodar

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

A documentação interativa fica em <http://localhost:8000/docs>.

## Endpoints

### `POST /ocr` — upload multipart

```bash
curl -F "file=@captcha.png" http://localhost:8000/ocr
# {"text":"x7kp2","elapsed_ms":4.1}
```

### `POST /ocr/base64` — JSON

Aceita base64 puro ou data URL (`data:image/png;base64,...`).

```bash
curl -H "Content-Type: application/json" \
     -d "{\"image\": \"$(base64 -w0 captcha.png)\"}" \
     http://localhost:8000/ocr/base64
```

### `POST /ocr/batch` — várias imagens de uma vez (até 16)

```bash
curl -F "files=@a.png" -F "files=@b.png" http://localhost:8000/ocr/batch
# {"texts":["x7kp2","9fgh3"],"elapsed_ms":7.9}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","device":"cpu"}
```

Erros: `400` (imagem ou base64 inválido), `401` (API key), `413` (imagem grande demais).

## Chamando do seu serviço (Python)

```python
import requests

with open("captcha.png", "rb") as f:
    r = requests.post("http://localhost:8000/ocr", files={"file": f}, timeout=30)
r.raise_for_status()
print(r.json()["text"])
```

## Docker

O modelo vai dentro da imagem:

```bash
docker build -t ocr-captcha .
docker run -p 8000:8000 ocr-captcha
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

## Medir a precisão

Coloque captchas numa pasta com o **texto certo como nome do arquivo**
(`u4ep.png`, `cma5c.png`; para repetir um texto, `u4ep_2.png`) e rode:

```bash
python scripts/evaluate.py captchas/rotulados
```

Mostra a taxa de acerto, o CER (erro por caractere) e o que o modelo leu em cada imagem.
`--models` compara vários de uma vez (outro `.pt`, `easyocr`, `easyocr:full` ou `ddddocr`,
os dois últimos instalados por `requirements-dev.txt`):

```bash
python scripts/evaluate.py captchas/rotulados --models models/crnn.pt easyocr ddddocr
```

## Rotular captchas novos

Coloque as imagens em `captchas/revisar/` e rode:

```bash
python scripts/prelabel.py captchas/revisar
```

Cada imagem é renomeada para o texto que o modelo leu, e `captchas/conferir.png` mostra as
leituras das menos confiáveis para as mais confiáveis. Corrija o nome dos arquivos errados
e mova-os para `captchas/rotulados/`. Rodar de novo não mexe nos arquivos que já têm nome de
rótulo.

## Treinar

### Captchas sintéticos

`scripts/synth_captcha.py` gera captchas no mesmo estilo dos reais, com o texto certo como
nome do arquivo. O treino gera os seus na hora; este comando serve para ver como eles ficam:

```bash
python scripts/synth_captcha.py dados/sinteticos --count 500
```

### Continuar o treino com os captchas rotulados

```bash
python scripts/train_crnn.py --real captchas/rotulados --holdout 10 \
  --init models/crnn.pt --out models/crnn-novo.pt
```

- `--real` mistura os captchas reais com os sintéticos (30% de cada lote, com pequenas
  variações de posição, rotação e ruído).
- `--holdout 10` separa 10 deles para medir cada época; ficam fora do treino e são sempre os
  mesmos, então os números de rodadas diferentes são comparáveis.
- A cada época mostra os acertos e o CER na validação e salva o melhor modelo em `--out`.
  Se ele for melhor que o atual, copie-o para `models/crnn.pt`.
- Na CPU (4 núcleos) cada época leva ~3 min. Sem `--init`, treina do zero: use
  `--lr 1e-3 --epochs 30`, porque as primeiras ~6 épocas são só de aquecimento.

Quanto mais captchas rotulados em `captchas/rotulados`, melhor o modelo e mais confiável
a medição.

## Observações

- Imagens com transparência são achatadas sobre fundo branco antes da inferência. PNGs em
  tons de cinza de 16 bits são convertidos para 8 bits antes disso.
- O modelo só conhece letras minúsculas e dígitos, no estilo de `captchas/rotulados`.
  Em captchas de outro estilo ele erra; nesse caso é preciso treinar com exemplos deles.
