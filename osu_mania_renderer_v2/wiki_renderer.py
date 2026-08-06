"""Back-compat shim — the mania compositor moved to ``render/compositor.py``.

Kept so the bot/worker entrypoint
``python -m osu_mania_renderer_v2.wiki_renderer`` and any lingering
``from osu_mania_renderer_v2.wiki_renderer import X`` keep working while the
"wiki" naming is retired. Internal callers now import from
``osu_mania_renderer_v2.render.compositor`` directly; delete this shim once the
bot/worker command has been switched over (tracked in the mania v3 reorg).
"""
from osu_mania_renderer_v2.render.compositor import (  # noqa: F401
    ELEMENTS,
    RENDER_ORDER,
    ElementSpec,
    RenderError,
    SkinIni,
    SkinPair,
    VariableSpec,
    register_element,
    render,
)

if __name__ == "__main__":
    import importlib

    importlib.import_module("osu_mania_renderer_v2.render.compositor")._cli()
