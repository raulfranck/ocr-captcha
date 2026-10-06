# OCR Captcha API

API HTTP que recebe a imagem de um captcha e devolve o texto, usando o modelo
[`anuashok/ocr-captcha-v3`](https://huggingface.co/anuashok/ocr-captcha-v3)
(TrOCR ajustado a partir de `microsoft/trocr-base-printed`, ~1,3 GB, CER ≈ 1,4%).

## Requisitos

- Python 3.10 ou superior
- ~3 GB livres (modelo + PyTorch)
- GPU é opcional: na CPU cada captcha leva por volta de 0,2 a 1 s.

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

## 2. Baixar o modelo

```bash
python scripts/download_model.py
```

Os arquivos ficam em `models/ocr-captcha-v3/`. A partir daí a API funciona offline.

Se aparecer `429 Too Many Requests`, o Hugging Face está limitando o acesso anônimo.
O script espera e tenta de novo sozinho, e numa nova execução pula o que já baixou.
Para um limite bem maior, crie um token de leitura (grátis) em
<https://huggingface.co/settings/tokens> e rode:

```bash
# Git Bash / Linux / macOS
HF_TOKEN=hf_xxx python scripts/download_model.py
# PowerShell
$env:HF_TOKEN="hf_xxx"; python scripts/download_model.py
```
Se você pular este passo, o modelo é baixado automaticamente para o cache do
Hugging Face na primeira vez que a API subir.

## 3. Configurar (opcional)

```bash
cp .env.example .env
```

| Variável          | Padrão                    | O que faz                                                  |
|-------------------|---------------------------|------------------------------------------------------------|
| `MODEL_PATH`      | `./models/ocr-captcha-v3` | Pasta local do modelo ou ID no Hugging Face                |
| `DEVICE`          | `auto`                    | `cpu`, `cuda`, `mps` ou `auto`                             |
| `NUM_BEAMS`       | `2`                       | Beam search; 1 é mais rápido, 2 é o valor do treino        |
| `PREPROCESS`      | `none`                    | Limpeza antes do modelo: `none`, `median` (tira o ruído de pontos) ou `median_bin` |
| `MAX_IMAGE_BYTES` | `2000000`                 | Tamanho máximo do upload                                   |
| `API_KEY`         | vazio                     | Se definido, exige o header `X-API-Key` em todas as chamadas |

## 4. Rodar

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use **um worker só** (o padrão): cada worker carrega uma cópia do modelo na memória.
A documentação interativa fica em <http://localhost:8000/docs>.

## Endpoints

### `POST /ocr` — upload multipart

```bash
curl -F "file=@captcha.png" http://localhost:8000/ocr
# {"text":"X7kP2","elapsed_ms":312.4}
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
# {"texts":["X7kP2","9fGh3"],"elapsed_ms":540.1}
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

```bash
python scripts/download_model.py          # baixa o modelo uma vez no host
docker build -t ocr-captcha .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" ocr-captcha
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes usam um OCR falso, então rodam sem o modelo baixado.

## Medir a precisão

Coloque captchas numa pasta com o **texto certo como nome do arquivo**
(`u4ep.png`, `cma5c.png`; para repetir um texto, `u4ep_2.png`) e rode:

```bash
python scripts/evaluate.py captchas/rotulados
```

O script testa cada combinação de `PREPROCESS` e `NUM_BEAMS` (padrão: 1 e 2) e mostra a taxa de
acerto, o CER (erro por caractere) e o que o modelo leu em cada imagem. Coloque a
melhor combinação no `.env`. Quanto mais imagens rotuladas, mais confiável a
comparação: 30 ou mais já dão uma boa ideia.

## Diagnóstico

Se a API devolver um texto sem sentido (por exemplo `.com`), rode:

```bash
python scripts/diagnose.py caminho/captcha.png
```

Ele mostra se o modelo carregou completo, o formato da imagem e o que o modelo lê,
e salva ao lado da imagem a versão que o modelo recebe (`debug_*.png`).

## Observações

- Imagens com transparência são achatadas sobre fundo branco antes da inferência,
  como no exemplo do model card. PNGs em tons de cinza de 16 bits são convertidos
  para 8 bits antes disso.
- O modelo foi treinado num estilo específico de captcha (veja as imagens no model
  card). Em captchas muito diferentes a precisão cai; nesse caso o caminho é fazer
  fine-tuning com exemplos rotulados do seu captcha.
