# Suhina modeling

Research tools for modelling [Suhina](https://github.com/suhina-daw/suhina-daw)'s built-in devices
against captured references: fits of a device's model, measurements, comparison figures and
level-matched listening packages. Nothing here runs in Suhina. What the research finds is written
down in the main repository's `docs/spec/` before it is implemented there.

| Folder | Device |
|---|---|
| [`reverb/`](reverb/README.md) | the diffused FDN reverb research (model and parity checks in the main repository, `scripts/reverb/model/`) |

A shared, device-agnostic toolkit is proposed in
[`docs/toolkit-design.md`](docs/toolkit-design.md); it is not built yet.

## Rules

The main repository's [AGENTS.md](https://github.com/suhina-daw/suhina-daw/blob/main/AGENTS.md)
applies here as well, in particular:

- **Clean-room.** Knowledge comes from observing a device's output with self-authored stimuli and
  from public documentation. Captures of other products' output stay on the machine that made
  them: never commit audio captures, listening packages or other products' files. Scripts read
  them through environment variables (`SUHINA_*`).
- **No other products' names** in scripts, paths, data or results.
- **Licence header** on every source file; the code is GPL-3.0-or-later ([LICENSE](LICENSE)).

## Setup

```sh
python -m venv .venv
pip install -r requirements.txt
```

Each folder's README lists the environment variables it reads.
