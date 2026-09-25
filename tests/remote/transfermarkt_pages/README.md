# Recorded Transfermarkt pages

Gzipped copies of the four pages `remote/transfermarkt.py` parses, fetched on
31 August 2026:

| File | Page | Parsed by |
| --- | --- | --- |
| `premier_league_2025.html.gz` | `/premier-league/startseite/wettbewerb/GB1/plus/?saison_id=2025` | `get_teams_for_season` |
| `player_verletzungen.html.gz` | `/kyle-walker/verletzungen/spieler/95424` | `get_player_injuries` |
| `player_ausfaelle.html.gz` | `/kyle-walker/ausfaelle/spieler/95424` | `get_player_suspensions` |
| `player_transfers.json.gz` | `/ceapi/transferHistory/list/95424` | `get_player_transfers` |

Kyle Walker because one player's pages cover every case between them: injuries
with and without a games-missed count, a suspension, a non-injury absence in a
cup competition, and transfers in and out of the league.

`tests/remote/test_transfermarkt_parsing.py` reads them in place of a request. The
scrape catches parse failures per player, so without these a change to
Transfermarkt's markup would show up only as an `absences_xxyy.csv` missing a kind
of absence.

To refresh one, fetch the page and `gzip -9` it into place. Keep them gzipped -
the four together are 400KB of HTML and 60KB compressed.
