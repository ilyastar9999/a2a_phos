# Moshi-RU on Apple Metal - convenience targets.
# All targets assume the bundled virtualenv at .venv (see requirements.txt).

PY ?= .venv/bin/python
SIZE ?= small
STEPS ?= 1500
DATA ?= data/toy/codes
OUT ?= runs/$(SIZE)
# batch 8 x 128 frames + a slow warmup is the recipe that actually fits the
# tiny datasets; batch 4 x 96 stalls near ln(2048) and never memorises (see README).
BATCH ?= 8
SEQ ?= 128
WARMUP ?= 100

.PHONY: help doctor cli cli-mic toy-data train train-qlora encode transcribe speechru test clean

help:
	@echo "make doctor                 # check Metal + memory + toolchain"
	@echo "make cli IN=in.wav OUT=out.wav   # speech-to-speech (7B q4)"
	@echo "make cli-mic                # interactive microphone (7B q4)"
	@echo "make toy-data               # build a small Russian TTS dataset"
	@echo "make train DATA=data/toy/codes SIZE=nano STEPS=400"
	@echo "             # BATCH=8 SEQ=128 WARMUP=100 are the defaults that fit"
	@echo "make train-qlora            # QLoRA fine-tune of the 4-bit 7B"
	@echo "make speechru               # download + segment a SpeechRu slice"
	@echo "make encode                 # wav -> Moshi codes"
	@echo "make transcribe             # mlx-whisper alignments"
	@echo "make test                   # delay/undelay equivalence test"

doctor:
	$(PY) scripts/doctor.py

cli:
	$(PY) cli/moshi_ru.py --input $(IN) --output $(OUT) --show-text

cli-mic:
	$(PY) cli/moshi_ru.py --mic

toy-data:
	$(PY) scripts/make_toy_dataset.py --out data/toy --repeats 8

train:
	$(PY) train/train_mlx.py --size $(SIZE) --data-dir $(DATA) --out $(OUT) --steps $(STEPS) \
		--batch-size $(BATCH) --seq-len $(SEQ) --warmup $(WARMUP)

train-qlora:
	$(PY) train/train_mlx.py --size 7b --model q4 --data-dir $(DATA) --out runs/qlora \
		--steps $(STEPS) --batch-size 1 --seq-len 32 --lora-rank 8

speechru:
	$(PY) scripts/prepare_dataset.py --limit-gb 0.6 --max-clips 600 --clip-seconds 20 --stereo

encode:
	$(PY) scripts/encode_codes.py --data-dir data/speech-ru/wav --out data/speech-ru/codes \
		--other same --workers 2 --nice 10

transcribe:
	$(PY) scripts/transcribe.py --data-dir data/speech-ru/wav \
		--model mlx-community/whisper-medium-mlx --language ru

test:
	$(PY) tests/test_delay_equivalence.py

clean:
	rm -rf runs/* data/toy/codes/*.npz
