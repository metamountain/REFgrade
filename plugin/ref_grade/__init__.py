"""REF Grade – Krita-Plugin-Einstiegspunkt."""

from krita import Krita

from .krita_extension import RefGradeExtension

Krita.instance().addExtension(RefGradeExtension(Krita.instance()))
