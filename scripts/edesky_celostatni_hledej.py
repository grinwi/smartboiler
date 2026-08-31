#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edesky_celostatni_hledej.py — vyhledá dokumenty napříč VŠEMI úředními
deskami na edesky.cz (bez omezení na jednu obec/desku), pro víc tvarů
hledaného výrazu najednou, a spojí výsledky do jednoho přehledu.

Na rozdíl od edesky_fulltext_hledej.py (který stahuje a ručně prohledává
plný text VŠECH dokumentů jedné desky, protože API nemá "list vše" bez
klíčového slova) tady spoléháme na vlastní fulltextové vyhledávání API
(search_with=es prohledává i obsah dokumentu — viz apiary.apib) a jen
proženeme víc tvarů jména/příjmení, protože skloňování nemusí být
rozpoznáno spolehlivě pro každé jméno.

Použití:
    export EDESKY_API_KEY=VAS_KLIC
    python3 edesky_celostatni_hledej.py \\
        --words "Výleta,Výlety,Výletovi,Výletu,Výletou,Výletem" \\
        --created-from 2000-01-01
"""

import argparse
import os
import sys

from edesky_hledej import stahni_vse, _first


def _dedup(dokumenty):
    """Odstraní duplicity (stejný dokument se může trefit na víc hledaných tvarů)."""
    videno = set()
    unikatni = []
    for d in dokumenty:
        klic = _first(d, ["edesky_id", "id"]) or _first(d, ["edesky_url", "url"])
        if klic and klic in videno:
            continue
        if klic:
            videno.add(klic)
        unikatni.append(d)
    return unikatni


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Celostátní vyhledání dokumentů napříč všemi deskami edesky.cz.")
    p.add_argument("--api-key", default=os.environ.get("EDESKY_API_KEY"))
    p.add_argument("--words", required=True,
                   help="hledaná slova/tvary oddělené čárkou, např. 'Výleta,Výlety,Výletovi'")
    p.add_argument("--created-from", default="2000-01-01")
    p.add_argument("--search-with", default="es", choices=["es", "sql"])
    args = p.parse_args(argv)

    if not args.api_key:
        p.error("chybí API klíč — zadej --api-key nebo nastav EDESKY_API_KEY")

    slova = [s.strip() for s in args.words.split(",") if s.strip()]
    if not slova:
        p.error("--words musí obsahovat alespoň jedno slovo")

    vsechny_nalezy = []
    for slovo in slova:
        print("Hledám tvar %r napříč všemi deskami (created_from=%s)..." %
              (slovo, args.created_from), file=sys.stderr)
        dokumenty, chyba = stahni_vse(slovo, args.api_key, None,
                                       args.search_with, "date", args.created_from)
        if chyba:
            print("  CHYBA pro %r: %s" % (slovo, chyba), file=sys.stderr)
        print("  -> %d výsledků" % len(dokumenty), file=sys.stderr)
        for d in dokumenty:
            d.setdefault("_shoda_na_tvar", slovo)
        vsechny_nalezy.extend(dokumenty)

    unikatni = _dedup(vsechny_nalezy)

    print("\n" + "=" * 60)
    if not unikatni:
        print("Nic nenalezeno napříč žádnou deskou pro žádný ze zadaných tvarů: %s"
              % ", ".join(slova))
        return 0

    print("Nalezeno %d unikátních dokumentů napříč celým edesky.cz:\n" % len(unikatni))
    for d in unikatni:
        nazev = _first(d, ["name", "title"]) or "(bez názvu)"
        deska = _first(d, ["dashboard_name"])
        datum = _first(d, ["created_at", "edited_date", "created_date", "date"])
        url = _first(d, ["edesky_url", "url"])
        tvar = d.get("_shoda_na_tvar", "")
        print("- %s" % nazev)
        if deska:
            print("  deska: %s" % deska)
        if datum:
            print("  datum: %s" % datum)
        if url:
            print("  odkaz: %s" % url)
        if tvar:
            print("  nalezeno přes tvar: %s" % tvar)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
