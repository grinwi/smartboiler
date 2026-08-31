#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edesky_fulltext_hledej.py — stáhne VŠECHNY dokumenty dané desky edesky.cz
a sám prohledá jejich plný text (edesky_text_url) na zadaná slova.

Proč: API nemá endpoint pro "vypsat všechny dokumenty desky bez klíčového
slova" (keywords je povinný parametr). Proto se dokumenty desky získají
dotazem na široké klíčové slovo (typicky název obce, který je v hlavičce/
patičce prakticky každého úředního dokumentu z té desky) a search_with=es
(hledá i v obsahu). Nad staženými dokumenty se pak nezávisle na relevanci
edesky.cz stáhne holý extrahovaný text (.txt) a ručně se v něm hledají
zadaná slova — tak se ověří i případy, které by ES vyhledávání minulo.

Použití:
    export EDESKY_API_KEY=VAS_KLIC
    python3 edesky_fulltext_hledej.py \\
        --dashboard-id 1061 \\
        --broad-keywords "Bílovice" \\
        --words "Výleta,Výlety,Výletovi,Výletu,Výletou,Výletem" \\
        --created-from 2000-01-01
"""

import argparse
import os
import sys
import time

from edesky_hledej import stahni, stahni_vse, _first

TEXT_TIMEOUT_S = 30
SLEEP_MEZI_DOTAZY_S = 0.2  # slušnost k serveru edesky.cz


def _dedup(dokumenty):
    """Odstraní duplicity (stejný dokument se může objevit vícekrát mezi stránkami)."""
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


def _najdi_useky(text, slovo, kontext=60):
    """Vrátí seznam úryvků textu kolem každého výskytu slova (case-insensitive)."""
    useky = []
    text_l = text.lower()
    slovo_l = slovo.lower()
    start = 0
    while True:
        i = text_l.find(slovo_l, start)
        if i == -1:
            break
        zac = max(0, i - kontext)
        kon = min(len(text), i + len(slovo) + kontext)
        useky.append(text[zac:kon].replace("\n", " ").strip())
        start = i + len(slovo)
    return useky


def main(argv=None):
    p = argparse.ArgumentParser(description="Fulltextové prohledání všech dokumentů desky.")
    p.add_argument("--api-key", default=os.environ.get("EDESKY_API_KEY"))
    p.add_argument("--dashboard-id", required=True, help="ID úřední desky")
    p.add_argument("--broad-keywords", required=True,
                   help="široký výraz pokrývající prakticky všechny dokumenty desky, "
                        "např. název obce")
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

    print("Stahuji seznam dokumentů desky %s (broad-keywords=%r, created_from=%s)..."
          % (args.dashboard_id, args.broad_keywords, args.created_from), file=sys.stderr)
    dokumenty, chyba = stahni_vse(args.broad_keywords, args.api_key, args.dashboard_id,
                                   args.search_with, "date", args.created_from)
    if chyba:
        print("CHYBA při stahování seznamu: %s" % chyba, file=sys.stderr)
        if not dokumenty:
            return 1

    dokumenty = _dedup(dokumenty)
    print("Nalezeno %d unikátních dokumentů k prohledání.\n" % len(dokumenty), file=sys.stderr)

    nalezeno = []
    chybely_text = 0
    for i, d in enumerate(dokumenty, 1):
        nazev = _first(d, ["name", "title"]) or "(bez názvu)"
        text_url = _first(d, ["edesky_text_url"])
        edesky_url = _first(d, ["edesky_url", "url"])
        datum = _first(d, ["created_at", "edited_date", "created_date", "date"])

        if not text_url:
            chybely_text += 1
            continue

        text, chyba_t = stahni(text_url)
        time.sleep(SLEEP_MEZI_DOTAZY_S)
        if chyba_t or text is None:
            print("  [%d/%d] %s — nelze stáhnout text (%s)" %
                  (i, len(dokumenty), nazev, chyba_t), file=sys.stderr)
            continue

        zasahy = {}
        for slovo in slova:
            useky = _najdi_useky(text, slovo)
            if useky:
                zasahy[slovo] = useky

        if zasahy:
            nalezeno.append((nazev, datum, edesky_url, zasahy))
            print("  [%d/%d] SHODA: %s" % (i, len(dokumenty), nazev), file=sys.stderr)

    print("\n" + "=" * 60)
    if not nalezeno:
        print("Ve fulltextu žádného z %d dokumentů nebyla nalezena žádná ze zadaných forem: %s"
              % (len(dokumenty), ", ".join(slova)))
        if chybely_text:
            print("(%d dokumentů nemělo edesky_text_url, text.gz nebyl k dispozici)" % chybely_text)
        return 0

    print("Nalezeno %d dokumentů obsahujících hledaná slova:\n" % len(nalezeno))
    for nazev, datum, url, zasahy in nalezeno:
        print("- %s" % nazev)
        if datum:
            print("  datum: %s" % datum)
        if url:
            print("  odkaz: %s" % url)
        for slovo, useky in zasahy.items():
            print("  shoda '%s' (%d×):" % (slovo, len(useky)))
            for u in useky[:3]:
                print("    …%s…" % u)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
