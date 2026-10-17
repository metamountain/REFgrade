"""REFgrade – Krita-Plugin-Einstiegspunkt."""

from .version import __version__  # noqa: F401  (re-export)

try:
    from krita import Krita
except ImportError:
    Krita = None  # außerhalb von Krita (CLI: python -m ref_grade.runtime)

if Krita is not None:
    from .krita_extension import RefGradeExtension
    Krita.instance().addExtension(RefGradeExtension(Krita.instance()))
