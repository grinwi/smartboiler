#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edesky_region_hledej.py — fulltextově prohledá desky VŠECH obcí spadajících
pod stejného administrativního "rodiče" (okres/ORP) jako zadaná kotevní deska.

API nemá parametr "najdi mi okolí desky X" ani "najdi rodiče desky X" přímo —
proto se rodič dohledá nepřímo: stáhne se seznam desek celého kraje
(include_subordinated=1) a v něm se najde záznam kotevní desky, který už
obsahuje parent_id/parent_name. Pak se seznam desek pod tímto rodičem stáhne
znovu (include_subordinated=1) — to je hledané "okolí" — a nad každou takto
nalezenou deskou se spustí stejné fulltextové prohledání jako v
edesky_fulltext_hledej.py (broad-keywords = název dané obce).

Použití:
    export EDESKY_API_KEY=VAS_KLIC
    python3 edesky_region_hledej.py \\
        --anchor-dashboard-id 1061 \\
        --words "Výleta,Výlety,Výletovi,Výletu,Výletou,Výletem" \\
        --created-from 2000-01-01
"""

import argparse
import os
import sys

from edesky_hledej import stahni_desky, _first
from edesky_fulltext_hledej import sken_desku

JIHOMORAVSKY_KRAJ_ID = 32  # zjištěno z https://edesky.cz/desky/32-Jihomoravsk%C3%BD%20kraj
PREDPONY_NAZVU_OBCE = ("Statutární město ", "Městská část ", "MČ ", "Městys ",
                        "Město ", "Obec ")


def _obecny_nazev(nazev_desky):
    """Ořeže administrativní předponu ('Obec', 'Město'...) pro použití jako broad-keywords."""
    for predpona in PREDPONY_NAZVU_OBCE:
        if nazev_desky.startswith(predpona):
            nazev_desky = nazev_desky[len(predpona):]
            break
    return nazev_desky.split("(")[0].strip() or nazev_desky


def najdi_rodice(anchor_id, api_key, kraj_id):
    """Najde parent_id/parent_name kotevní desky prohledáním desek celého kraje."""
    desky, chyba = stahni_desky(api_key, id=kraj_id, include_subordinated=1)
    if chyba:
        return None, None, chyba
    anchor_id = str(anchor_id)
    for d in desky:
        if _first(d, ["edesky_id", "id"]) == anchor_id:
            rodic_id = _first(d, ["parent_id"])
            rodic_name = _first(d, ["parent_name"])
            if not rodic_id:
                return None, None, "kotevní deska %s nemá v datech parent_id" % anchor_id
            return rodic_id, rodic_name, None
    return None, None, "kotevní deska %s nenalezena mezi deskami kraje %s" % (anchor_id, kraj_id)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fulltextové prohledání desek okolních obcí (stejný admin. rodič jako kotevní deska).")
    p.add_argument("--api-key", default=os.environ.get("EDESKY_API_KEY"))
    p.add_argument("--anchor-dashboard-id", default="1061",
                   help="deska, jejíž okolí (obce se stejným rodičem) se má prohledat")
    p.add_argument("--kraj-id", type=int, default=JIHOMORAVSKY_KRAJ_ID,
                   help="ID kraje, ve kterém se dohledává rodič kotevní desky")
    p.add_argument("--words", required=True,
                   help="hledaná slova/tvary oddělené čárkou")
    p.add_argument("--created-from", default="2000-01-01")
    p.add_argument("--search-with", default="es", choices=["es", "sql"])
    p.add_argument("--max-desek", type=int, default=60,
                   help="pojistka: kolik desek okolí nejvýš prohledat")
    args = p.parse_args(argv)

    if not args.api_key:
        p.error("chybí API klíč — zadej --api-key nebo nastav EDESKY_API_KEY")

    slova = [s.strip() for s in args.words.split(",") if s.strip()]
    if not slova:
        p.error("--words musí obsahovat alespoň jedno slovo")

    print("Hledám administrativního rodiče kotevní desky %s (v kraji %s)..." %
          (args.anchor_dashboard_id, args.kraj_id), file=sys.stderr)
    rodic_id, rodic_name, chyba = najdi_rodice(args.anchor_dashboard_id, args.api_key, args.kraj_id)
    if chyba:
        print("CHYBA: %s" % chyba, file=sys.stderr)
        return 1
    print("Rodič kotevní desky: %s (id=%s)\n" % (rodic_name, rodic_id), file=sys.stderr)

    print("Stahuji seznam obcí spadajících pod %s..." % rodic_name, file=sys.stderr)
    podrizene, chyba = stahni_desky(args.api_key, id=rodic_id, include_subordinated=1)
    if chyba:
        print("CHYBA: %s" % chyba, file=sys.stderr)
        return 1

    obce = []
    videno = set()
    for d in podrizene:
        eid = _first(d, ["edesky_id", "id"])
        nazev = _first(d, ["name"])
        if not eid or not nazev or eid in videno:
            continue
        videno.add(eid)
        obce.append((eid, nazev))

    print("Nalezeno %d desek pod %s." % (len(obce), rodic_name), file=sys.stderr)
    if len(obce) > args.max_desek:
        print("Omezuji na prvních %d (--max-desek)." % args.max_desek, file=sys.stderr)
        obce = obce[:args.max_desek]
    print(file=sys.stderr)

    vysledky = []
    celkem_dokumentu = 0
    for i, (eid, nazev) in enumerate(obce, 1):
        broad = _obecny_nazev(nazev)
        print("[%d/%d] %s (deska %s, broad-keywords=%r)" % (i, len(obce), nazev, eid, broad),
              file=sys.stderr)
        nalezeno, pocet_dok, chybely_text, chyba_seznamu = sken_desku(
            eid, broad, slova, args.api_key, args.created_from, args.search_with,
            stderr_prefix="  ")
        celkem_dokumentu += pocet_dok
        if chyba_seznamu:
            print("    CHYBA při stahování seznamu: %s" % chyba_seznamu, file=sys.stderr)
        print("  -> %d dokumentů prohledáno, %d shod" % (pocet_dok, len(nalezeno)), file=sys.stderr)
        for nazev_dok, datum, url, zasahy in nalezeno:
            vysledky.append((nazev, nazev_dok, datum, url, zasahy))

    print("\n" + "=" * 60)
    print("Prohledáno %d desek (okolí %s), celkem %d dokumentů.\n" %
          (len(obce), rodic_name, celkem_dokumentu))

    if not vysledky:
        print("Ve fulltextu žádné z prohledaných desek nebyla nalezena shoda pro: %s"
              % ", ".join(slova))
        return 0

    print("Nalezeno %d dokumentů se shodou:\n" % len(vysledky))
    for deska_nazev, nazev_dok, datum, url, zasahy in vysledky:
        print("- %s" % nazev_dok)
        print("  deska: %s" % deska_nazev)
        if datum:
            print("  datum: %s" % datum)
        if url:
            print("  odkaz: %s" % url)
        for slovo, useky in zasahy.items():
            print("  shoda '%s' (%d×): …%s…" % (slovo, len(useky), useky[0]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
