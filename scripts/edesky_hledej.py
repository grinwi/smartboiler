#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edesky_hledej.py — vyhledávání dokumentů v API portálu edesky.cz.

KISS: pouze standardní knihovna (urllib + xml.etree), žádné závislosti.

Použití:
    # klíč jako parametr
    python3 edesky_hledej.py --keywords "Výleta" --api-key VAS_KLIC

    # klíč z proměnné prostředí (bezpečnější — neuloží se do historie shellu)
    export EDESKY_API_KEY=VAS_KLIC
    python3 edesky_hledej.py --keywords "Výleta"

    # omezení na jednu desku (Bílovice nad Svitavou = 1061)
    python3 edesky_hledej.py --keywords "Výleta" --dashboard-id 1061

    # kompletní historie, ne jen nedávné dokumenty (bez created_from API
    # vrací jen nedávnou dobu)
    python3 edesky_hledej.py --keywords "Výleta" --created-from 2000-01-01

    # ověření parseru bez sítě
    python3 edesky_hledej.py --selftest

Poznámky k API (zdroj: https://edesky.cz/api a github.com/edesky/edesky_api):
  - endpoint:   https://edesky.cz/api/v1/documents
  - api_key:    povinný, získává se registrací na https://edesky.cz/api
  - keywords:   hledaný výraz
  - search_with: "es" hledá i ve skloňovaných tvarech a v obsahu dokumentu
                 (vhodné pro příjmení), "sql" hledá jen přesně v názvu
  - order:      "date" = řadit podle data
  - dashboard_id: volitelné omezení na konkrétní úřední desku
  - page:       API vrací max 200 dokumentů na stránku; skript automaticky
                stáhne a spojí VŠECHNY stránky, ne jen tu první
  - created_from: bez zadání API vrací jen nedávné dokumenty (přesné okno
                není zdokumentované); pro celou historii zadej staré datum

Přesný název parametru pro filtr desky ("dashboard_id") je odvozen; pokud by
API vrátilo chybu parametru, mrkněte do apiary.apib v repu edesky_api a hodnotu
upravte v konstantě PARAM_DASHBOARD níže.
"""

import argparse
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

API_URL = "https://edesky.cz/api/v1/documents"
PARAM_DASHBOARD = "dashboard_id"  # v případě potřeby uprav dle apiary.apib
TIMEOUT_S = 30
STRANKA_VELIKOST = 200  # dle apiary.apib vrací API max 200 dokumentů na stránku
MAX_STRANEK = 100  # pojistka proti nekonečnému stahování (max 20 000 dokumentů)


def sestav_url(keywords, api_key, dashboard_id=None, search_with="es", order="date",
               page=None, created_from=None):
    """Sestaví URL dotazu. Vrací hotovou adresu se správně zakódovanými parametry."""
    params = {
        "keywords": keywords,
        "api_key": api_key,
        "search_with": search_with,
        "order": order,
    }
    if dashboard_id:
        params[PARAM_DASHBOARD] = dashboard_id
    if page:
        params["page"] = page
    if created_from:
        params["created_from"] = created_from
    return API_URL + "?" + urllib.parse.urlencode(params)


def stahni(url):
    """Stáhne obsah URL. Vrací (text, None) při úspěchu, (None, chyba) při selhání."""
    req = urllib.request.Request(url, headers={"User-Agent": "edesky-hledej/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.read().decode("utf-8"), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "HTTP 401 — neplatný nebo chybějící api_key."
        return None, "HTTP %d — %s" % (e.code, e.reason)
    except urllib.error.URLError as e:
        return None, "Chyba sítě: %s" % e.reason


def stahni_vse(keywords, api_key, dashboard_id=None, search_with="es", order="date",
               created_from=None):
    """
    Stáhne a naparsuje VŠECHNY stránky výsledků (API vrací max 200 dokumentů
    na stránku). Vrací (seznam_dokumentů, None) při úspěchu, ([], chyba) při
    selhání. Stahování skončí, jakmile stránka vrátí méně než plnou dávku,
    nebo po dosažení MAX_STRANEK.

    Bez created_from API vrací jen dokumenty z nedávné doby (přesné výchozí
    okno není zdokumentované) — pro kompletní historii zadej created_from
    hodně starým datem, např. "2000-01-01".
    """
    vsechny = []
    for stranka in range(1, MAX_STRANEK + 1):
        url = sestav_url(keywords, api_key, dashboard_id, search_with, order,
                         page=stranka, created_from=created_from)
        text, chyba = stahni(url)
        if chyba:
            return vsechny, chyba
        try:
            dokumenty = parsuj(text)
        except ET.ParseError as e:
            return vsechny, "odpověď není platné XML (%s)" % e
        vsechny.extend(dokumenty)
        if len(dokumenty) < STRANKA_VELIKOST:
            break
    return vsechny, None


def _text(el):
    return (el.text or "").strip() if el is not None else ""


def parsuj(xml_text):
    """
    Naparsuje XML odpovědi na seznam slovníků.

    Parser je záměrně defenzivní — nespoléhá na přesné schéma. Z každého prvku,
    jehož tag končí na 'document', vytáhne atributy i texty přímých potomků,
    takže funguje, i kdyby edesky drobně změnilo strukturu.
    """
    root = ET.fromstring(xml_text)
    dokumenty = []
    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()  # odstraní případný namespace
        if tag != "document":
            continue
        zaznam = dict(el.attrib)  # atributy prvku <document ...>
        for dite in el:  # texty potomků (kdyby data byla v elementech)
            k = dite.tag.split("}")[-1]
            if k not in zaznam and _text(dite):
                zaznam[k] = _text(dite)
        dokumenty.append(zaznam)
    return dokumenty


def _first(d, klice):
    """Vrátí první neprázdnou hodnotu z daných klíčů (case-insensitive)."""
    lower = {k.lower(): v for k, v in d.items()}
    for k in klice:
        v = lower.get(k.lower())
        if v:
            return v
    return ""


def vypis(dokumenty):
    """Vytiskne dokumenty čitelně. Když nezná pole, vypíše syrové atributy."""
    if not dokumenty:
        print("Nic nenalezeno.")
        return
    print("Nalezeno dokumentů: %d\n" % len(dokumenty))
    for i, d in enumerate(dokumenty, 1):
        nazev = _first(d, ["name", "title", "nazev"]) or "(bez názvu)"
        datum = _first(d, ["edited_date", "created_date", "date", "edited_at", "datum"])
        url = _first(d, ["url", "link", "odkaz"])
        deska = _first(d, ["dashboard_name", "dashboard_id", "dashboard"])
        print("%2d. %s" % (i, nazev))
        if datum:
            print("    datum: %s" % datum)
        if deska:
            print("    deska: %s" % deska)
        if url:
            print("    odkaz: %s" % url)
        # cokoli navíc, ať nic neztratíme
        zbytek = {k: v for k, v in d.items()
                  if k.lower() not in {
                      "name", "title", "nazev", "url", "link", "odkaz",
                      "edited_date", "created_date", "date", "edited_at", "datum",
                      "dashboard_name", "dashboard_id", "dashboard"}}
        if zbytek:
            print("    další: %s" % zbytek)
        print()


def selftest():
    """Ověří parser na ukázkovém XML (bez sítě). Vrací 0 při úspěchu."""
    vzorek = """<?xml version="1.0" encoding="UTF-8"?>
    <documents>
      <document id="123" name="Rozhodnutí - stavební povolení - Výleta"
                url="https://edesky.cz/dokument/123"
                edited_date="2023-06-01" dashboard_id="1061"/>
      <document id="124" name="Veřejná vyhláška - Výleta, novostavba RD"
                url="https://edesky.cz/dokument/124"
                edited_date="2023-05-20" dashboard_id="1061"/>
    </documents>"""
    d = parsuj(vzorek)
    assert len(d) == 2, "čekány 2 dokumenty, dostal %d" % len(d)
    assert d[0]["name"].endswith("Výleta"), "špatně načtený název"
    assert d[0]["url"] == "https://edesky.cz/dokument/123", "špatně načtená URL"
    assert d[1]["dashboard_id"] == "1061", "špatně načtená deska"
    # kontrola i pro variantu s daty v potomcích (jiné možné schéma)
    vzorek2 = """<documents><document><name>Test</name>
                 <url>http://x</url></document></documents>"""
    d2 = parsuj(vzorek2)
    assert d2[0]["name"] == "Test" and d2[0]["url"] == "http://x", "parsování potomků selhalo"
    print("Self-test OK — parser funguje.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Vyhledávání dokumentů v edesky.cz API.")
    p.add_argument("--keywords", help="hledaný výraz, např. 'Výleta'")
    p.add_argument("--api-key", default=os.environ.get("EDESKY_API_KEY"),
                   help="API klíč (nebo proměnná EDESKY_API_KEY)")
    p.add_argument("--dashboard-id", help="volitelně omezit na desku, Bílovice n. Sv. = 1061")
    p.add_argument("--search-with", default="es", choices=["es", "sql"],
                   help="es = i skloňované + obsah (výchozí), sql = jen název")
    p.add_argument("--order", default="date", help="řazení (výchozí: date)")
    p.add_argument("--created-from",
                   help="jen dokumenty načtené po tomto datu (YYYY-MM-DD); "
                        "bez tohoto parametru API vrací jen nedávné dokumenty, "
                        "pro celou historii zadej např. 2000-01-01")
    p.add_argument("--url-only", action="store_true",
                   help="jen vypsat sestavenou URL, nedotazovat")
    p.add_argument("--selftest", action="store_true", help="ověřit parser bez sítě")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.keywords:
        p.error("chybí --keywords (nebo použij --selftest)")
    if not args.api_key:
        p.error("chybí API klíč — zadej --api-key nebo nastav EDESKY_API_KEY")

    if args.url_only:
        print(sestav_url(args.keywords, args.api_key, args.dashboard_id,
                         args.search_with, args.order, created_from=args.created_from))
        return 0

    dokumenty, chyba = stahni_vse(args.keywords, args.api_key, args.dashboard_id,
                                   args.search_with, args.order, args.created_from)
    if chyba:
        print("CHYBA: %s" % chyba, file=sys.stderr)
        if not dokumenty:
            return 1
        print("(zobrazuji alespoň částečně stažené výsledky)", file=sys.stderr)
    vypis(dokumenty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
