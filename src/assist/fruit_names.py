"""Spoken names for every fruit class: English singular / plural and the common Filipino name.

Covers both models: the 6-class v1 model (Apple, Banana, ...) and the 30-class v2 model
(ambarella_fruit, avocado, ...). Filipino names are given only where a common one exists.
"""
from __future__ import annotations

# class name -> (singular, plural, Filipino name or None)
FRUIT_NAMES: dict[str, tuple[str, str, str | None]] = {
    # v1 (6 classes)
    "Apple": ("apple", "apples", "mansanas"),
    "Banana": ("banana", "bananas", "saging"),
    "Grape": ("grape bunch", "grape bunches", "ubas"),
    "Orange": ("orange", "oranges", "dalandan"),
    "Pineapple": ("pineapple", "pineapples", "pinya"),
    "Watermelon": ("watermelon", "watermelons", "pakwan"),
    # combined 31-class model: apple and grape come from v1 (lower-case names)
    "apple": ("apple", "apples", "mansanas"),
    "grape": ("grape bunch", "grape bunches", "ubas"),
    # v2 (30 classes, Roboflow "Fruit" by trucly)
    "ambarella_fruit": ("ambarella", "ambarellas", None),
    "avocado": ("avocado", "avocados", "abokado"),
    "banana": ("banana", "bananas", "saging"),
    "burmese_grape": ("Burmese grape", "Burmese grapes", None),
    "carambola": ("star fruit", "star fruits", "balimbing"),
    "cempedak": ("cempedak", "cempedaks", None),
    "coconut": ("coconut", "coconuts", "niyog"),
    "custard_apple": ("custard apple", "custard apples", "anonas"),
    "dragon_fruit": ("dragon fruit", "dragon fruits", None),
    "durian": ("durian", "durians", None),
    "grapefruit": ("grapefruit", "grapefruits", None),
    "guava": ("guava", "guavas", "bayabas"),
    "lanzones": ("lanzones", "lanzones", "lansones"),
    "lychee": ("lychee", "lychees", None),
    "mango": ("mango", "mangoes", "mangga"),
    "mangosteen": ("mangosteen", "mangosteens", "manggustin"),
    "melon": ("melon", "melons", None),
    "orange": ("orange", "oranges", "dalandan"),
    "papayas": ("papaya", "papayas", None),
    "pear": ("pear", "pears", "peras"),
    "persimmon": ("persimmon", "persimmons", None),
    "pineapple": ("pineapple", "pineapples", "pinya"),
    "pomegranate": ("pomegranate", "pomegranates", "granada"),
    "rambutan": ("rambutan", "rambutans", None),
    "rose_apple": ("rose apple", "rose apples", "makopa"),
    "sapodilla": ("sapodilla", "sapodillas", "chico"),
    "star_apple": ("star apple", "star apples", "kaimito"),
    "tangerines": ("tangerine", "tangerines", "dalanghita"),
    "watermelon": ("watermelon", "watermelons", "pakwan"),
    "wood_apple": ("wood apple", "wood apples", None),
}


def names_for(cls: str) -> tuple[str, str, str | None]:
    """(singular, plural, Filipino) for a class; unknown classes get a readable fallback."""
    if cls in FRUIT_NAMES:
        return FRUIT_NAMES[cls]
    word = cls.replace("_", " ").lower()
    return word, word + "s", None


def display_name(cls: str, filipino: bool = True) -> str:
    """'Mango (mangga)' for tables and labels."""
    singular, _, fil = names_for(cls)
    name = singular[:1].upper() + singular[1:]
    return f"{name} ({fil})" if filipino and fil and fil.lower() != singular.lower() else name
