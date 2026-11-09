# Honor an explicit AV1 VAAPI bitrate override

The published 0.1.38 mania engine's `_av1_codec_args` returns its VAAPI CQP
arguments before consulting `video_bitrate_override`. An API caller requesting
3,000,000 bps therefore gets `-rc_mode CQP -global_quality 90`, with no `-b:v`.
Its public CLI also has no `--video-bitrate` flag, although RenderOptions and
the render orchestrator already support passing an explicit override.

This fix adds the CLI flag (positive finite bps, decimal `k`/`M` suffixes) and
maps it to the existing `RenderOptions.video_bitrate_override`. AV1 VAAPI now
honors an explicit target with the existing published sibling-engine VBR recipe:
`-rc_mode VBR -b:v TARGET -maxrate 1.5*TARGET`, keeping AV1 Main and the same GOP.
Omitting the flag preserves CQP/global_quality 90 and its existing environment
override. H.264 preview/compact command arguments remain unchanged; the master
stays AV1. No default bitrate or user interface is added.

## Provenance

The base is exact published mania commit
`c74551f3398cab4fb5087090e4262774deba8e43`. Before editing, `cli.py`,
`render/encode.py`, `render/render.py`, `beatmap/models.py` and `__init__.py`
were byte-for-byte identical to those entries in the verified bundle 0.1.38.
Its 75,097,509-byte archive SHA256 is
`61e51e2c59ce6dd671eaa48018fcd623ecb2386094093e184a49c4f69467a72e`;
Ed25519 verifies against the contributor source's default public key. No bundle
was installed or executed. This establishes source identity, not installed
fleet state or the signing pin actually configured on any node.

The signed dispatcher appends unknown CLI flags to the selected engine command,
so it already forwards this explicit flag. Contributor source `4250b72` forces
encoder/device without adding a bitrate default. Companion contributor work
adds only mania `--video-bitrate` allowlist parity and argument preservation
tests; it adds no new UI or preset JSON field. Existing operator command
arguments can supply the flag.

## Validation and limits

`tests/test_av1_bitrate_override.py` first reproduced six failures against the
published baseline (ignored rate override and rejected CLI flags). The fixed
CLI, AV1 argument builder, no-override recipe and H.264 companion role pass the
new tests together with existing AV1/probe/CLI tests:

```
python -m pytest tests/test_av1_bitrate_override.py tests/test_av1_encode.py \
  tests/test_encoder_runtime.py tests/test_cli.py
```

The complete suite currently stops with 11 collection errors in legacy tests
importing removed module paths or removed exports (`test_beatmap`, `converter`,
`encode`, `gpu_renderer_background`, `gpu_renderer_hud`,
`gpu_renderer_playfield`, `judgments`, `models`, `mods`, `replay`,
`sv_integration`). These are a separate verification gate; the full suite is
not claimed green. The AV1 startup probe continues to test the existing default
CQP recipe; explicit-target VBR driver support still needs its own hardware run.

These are argument/control-flow proofs, not visual-quality measurements. The
reported AV1 blockiness still requires the same frame-aligned lossless pixel
input for current/candidate AV1 and H.264, representative four-mode content,
LPIPS distributions and other metrics, and the actual selected driver/backend.
A once-compressed source decoded once to FFV1 can measure added encoding error,
but excludes damage already in that source. No quality threshold, AMD-only
cause, “transparent” claim or new production quality default is established.
Publication requires fresh review, the paired client change where applicable,
and a newly built/signed bundle; no live or fleet setting was changed here.
