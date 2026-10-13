"""Registry für experimentelle und stabile Methoden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_REGISTRY: dict[str, "MethodInfo"] = {}


@dataclass(frozen=True)
class MethodInfo:
    name: str
    fn: Callable
    stable: bool


def register(name: str, stable: bool = False):
    """Dekorator: registriert eine Methode.

    Die dekorierte Funktion hat die Signatur:
        fit(image, reference, image_weights=None, reference_weights=None) -> Callable
    """
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = MethodInfo(name=name, fn=fn, stable=stable)
        return fn
    return decorator


def get(name: str) -> Callable:
    """Methode nach Name abrufen (die Fit-Funktion)."""
    if name not in _REGISTRY:
        raise KeyError(f"Unbekannte Methode: {name!r}. Verfügbar: {list(_REGISTRY.keys())}")
    return _REGISTRY[name].fn


def is_stable(name: str) -> bool:
    """True wenn Methode als stable registriert ist."""
    return _REGISTRY[name].stable


def list_all() -> list[str]:
    """Alle registrierten Methodennamen."""
    return list(_REGISTRY.keys())


def list_stable() -> list[str]:
    """Alle stable Methodennamen."""
    return [name for name, info in _REGISTRY.items() if info.stable]
