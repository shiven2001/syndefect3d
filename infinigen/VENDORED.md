# Vendored Infinigen dependencies

These directories are copied into the repo (not git submodules) so a plain `git clone` is enough to run `generate_indoors` and the defect pipeline.

| Path | Upstream | Pin |
|------|----------|-----|
| `infinigen/infinigen_gpl/` | [princeton-vl/infinigen_gpl](https://github.com/princeton-vl/infinigen_gpl) | `10c1d76` |
| `infinigen/OcMesher/` | [princeton-vl/OcMesher](https://github.com/princeton-vl/OcMesher) | `bb895a01` (v2.0) |
| `infinigen/assets/objects/creatures/parts/` | [princeton-vl/infinigen](https://github.com/princeton-vl/infinigen) `main` | — |
| `infinigen/assets/objects/creatures/insects/parts/` | same | — |

Licenses: `infinigen_gpl` is GPL-3.0; `OcMesher` is BSD-3-Clause (see each tree’s `LICENSE`). SynDefect3D is GPL-3.0.

To refresh vendored trees from upstream, use `scripts/bootstrap_vendored_deps.sh` (then remove any new `.git` folders before committing).
