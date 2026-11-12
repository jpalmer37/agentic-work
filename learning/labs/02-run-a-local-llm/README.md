# Lab 02: Run a local LLM, then remove it

## Goal

Install a real language model server, run an open-weight model on your
GPU, measure it, call it the way an application would, and then remove
every trace of it.

## What you will see

- A 3-billion-parameter model answering questions on your own machine,
  with nothing sent to the internet.
- GPU memory filling as the model loads and emptying when it unloads.
- Prefill running many times faster per token than decode.
- The same model answering through the standard OpenAI-style API.
- A clean machine afterwards.

## Time and hardware

About 30 minutes, including a download of about 2 GB. An NVIDIA GPU is
strongly recommended. It works on a CPU, slowly.

## Read first

- [Module 7: LLM architectures](../../07-llm-architectures.md), for
  base and chat models and parameter counts.
- [Module 8: Inference](../../08-inference-and-generation.md), for
  prefill, decode, the KV cache and quantization.
- [Module 13: Hosting](../../13-hosting-your-own-models.md), for the
  standard API and memory sizing.

## About the tools

**Ollama** is a local model runner. It downloads quantized models,
loads them onto your GPU and serves them over HTTP. It is built for one
person on one machine, which is exactly this lab. Lab 03 uses a server
built for many users.

**The model** is `llama3.2:3b`: Meta's Llama 3.2 with 3 billion
parameters, quantized to about 4 bits, so it takes about 2 GB. Its
licence allows this kind of personal use. Checked November 2026.

## Steps

### 1. Install Ollama

```bash
curl -fsSL \
  https://ollama.com/install.sh | sh
```

The installer copies the program, creates an `ollama` service user,
and starts a background service. It needs systemd, which you turned on
in Lab 00.

Check it is running:

```bash
ollama -v
systemctl status ollama
```

The status should say `active (running)`. Press `q` to leave it.

### 2. Watch the GPU

Open a **second** Ubuntu terminal and leave this running for the rest
of the lab:

```bash
watch -n 1 nvidia-smi
```

Note the memory used before you start, for example `1200MiB`.

### 3. Download and chat

Back in the first terminal:

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

You now have a chat prompt. Ask it something, and watch the memory jump
in the second terminal as the model loads. Type `/bye` to leave.

### 4. See what is loaded

```bash
ollama list
ollama ps
```

`list` shows models on disk. `ps` shows models in memory, how much
memory each uses, and whether it runs on the GPU. You want
`100% GPU`.

Ollama keeps a model loaded for 5 minutes after its last use, then
unloads it automatically.

### 5. Measure it

```bash
cd ~/agent-skills/learning/labs
source .venv/bin/activate
cd 02-run-a-local-llm
python llm_stats.py
```

The script makes four calls:

1. A short prompt, first time. Includes loading.
2. The same prompt again.
3. A long prompt of about 2,000 tokens.
4. A streaming call through the OpenAI-compatible API.

For each, it shows prefill and decode speed separately, like this
(numbers vary by machine):

```text
prefill   2414 tokens in 0.6 s
          = 4000 tokens/s
decode     120 tokens in 1.2 s
          =  100 tokens/s
```

### 6. Call it like an application

Any code written for the OpenAI API can use your local model by
changing the address. This is from `llm_stats.py`:

```python
from openai import OpenAI

client = OpenAI(
    base_url=
      "http://localhost:11434/v1",
    api_key="ollama",  # ignored
)
```

Try `curl` too:

```bash
curl localhost:11434/v1/models
```

### 7. Try a bigger model (optional)

```bash
ollama run llama3.1:8b
```

This one is about 5 GB. Compare the memory in `nvidia-smi`, the decode
speed from `llm_stats.py --model llama3.1:8b`, and the quality of its
answers.

## What to notice

**Memory matches the arithmetic.** A 3B model at about 4 bits needs
roughly `3 * 0.6 = 1.8 GB` for weights. `ollama ps` shows a little more:
the KV cache and working memory sit on top, exactly as
[Module 13](../../13-hosting-your-own-models.md) describes.

**Prefill is far faster per token than decode.** Often 20 to 50 times.
The prompt is processed in one parallel pass. The answer comes out one
token at a time. This is why output tokens cost more than input tokens.

**A longer prompt slows the first token, not the rest.** The long
prompt's prefill takes longer in total, but decode speed barely
changes. Time to first token depends on prompt length. Speed after that
depends on the model.

**Loading is a one-off.** The first call includes several seconds of
loading. The second does not. This is why servers keep models warm.

**Quantization made this possible.** At 16 bits the 8B model would need
about 16 GB, more than your GPU has. At 4 bits it fits with room to
spare.

**Nothing left your machine.** The prompt, the model and the answer
all stayed local. That is the first reason companies self-host.

## Spin down and clean up

Do these in order and watch `nvidia-smi` as you go.

**1. Unload the model from GPU memory.**

```bash
ollama stop llama3.2:3b
ollama ps
```

`ps` is now empty and GPU memory is back to where it started.

**2. Stop the server.**

```bash
sudo systemctl stop ollama
```

To stop it starting with WSL:

```bash
sudo systemctl disable ollama
```

Nothing is using the GPU or listening on port 11434 now. To bring it
back later: `sudo systemctl start ollama`.

**3. Delete the models.** If you plan to do Lab 04 soon, skip this
step: it uses the same server and model.

```bash
sudo systemctl start ollama
ollama rm llama3.2:3b
ollama rm llama3.1:8b
ollama list
sudo systemctl stop ollama
```

The models live in `/usr/share/ollama/.ollama/models`.

**4. Uninstall completely (optional).** These are Ollama's official
Linux uninstall steps:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm \
  /etc/systemd/system/ollama.service
sudo rm -r \
  $(which ollama | tr 'bin' 'lib')
sudo rm $(which ollama)
sudo userdel ollama
sudo groupdel ollama
sudo rm -r /usr/share/ollama
```

## Troubleshooting

**`ollama ps` shows CPU, or a CPU/GPU split.** The model did not fit
in GPU memory, or the GPU was not found. Close other GPU programs. Check
the service log for the GPU it detected:

```bash
journalctl -e -u ollama
```

**`Failed to connect to systemd`.** systemd is not running in WSL. See
step 4 of Lab 00. Or run `ollama serve` in its own terminal instead,
and press Ctrl+C to stop it.

**`connection refused` on port 11434.** The server is stopped. Run
`sudo systemctl start ollama`.

**Answers are slow and the GPU is idle.** WSL may not see the GPU. Run
`nvidia-smi` in WSL. If it fails, revisit Lab 00.

---

[← Lab 01: Train a tiny GPT](../01-train-a-tiny-gpt/README.md) ·
[Labs home](../README.md) ·
[Next: Lab 03, Serve like a product →](../03-serve-like-a-product/README.md)
