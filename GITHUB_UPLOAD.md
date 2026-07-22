# GitHub Upload Notes

The project is ready as a local git repository. Upload requires a GitHub account
and authentication.

## If GitHub CLI Is Available

```bash
gh auth login
gh repo create kimi-k3-streaming-lab --public --source . --remote origin --push
```

## If Using The Browser

1. Create or log into a GitHub account in the browser.
2. Create a new repository named `kimi-k3-streaming-lab`.
3. Keep it public or private as preferred.
4. Do not initialize with a README, because this project already has one.
5. Add the remote and push:

```bash
git remote add origin https://github.com/<your-username>/kimi-k3-streaming-lab.git
git branch -M main
git push -u origin main
```

## Repository Description

```text
C scaffold for a Colibri-style MoE expert-streaming engine targeting Kimi K3 once official config and weights are published.
```

## What To Mention Publicly

```text
This project is ready for model-independent engine work. It is waiting for the official Kimi K3 release artifacts: config.json, modeling code, tokenizer files, safetensors index, and license confirmation. Full weights should not be downloaded until those metadata gates pass.
```
