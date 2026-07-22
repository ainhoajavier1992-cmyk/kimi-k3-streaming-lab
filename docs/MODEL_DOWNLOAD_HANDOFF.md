# Model Download Handoff

Do not download Kimi K3 yet.

The current public K3 page does not expose the files needed for a real adapter:
`config.json`, modeling code, tokenizer files, safetensors index, or weights.
Until those appear, the engine can be developed and tested with synthetic MoE
fixtures only.

## Tell The User To Download When

Ask for the model download only after all of these are true:

- Kimi K3 has an official or trusted release page with real weights.
- `config.json` is available.
- `model.safetensors.index.json` or an equivalent weight map is available.
- Modeling code is available or the architecture is supported by upstream
  Transformers.
- The license permits local conversion and inference.

## First Download Should Be Metadata Only

The first pass should fetch only small files:

- `config.json`
- `generation_config.json`
- tokenizer files
- modeling/configuration Python files
- `model.safetensors.index.json`

Then run:

```bash
python3 tools/kimi_convert_skeleton.py --snapshot /path/to/kimi-k3-snapshot
```

If that reports `ready_for_tensor_mapping`, then a full download/conversion is
worth doing.
