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

These exist because there were no recorded pages before them. Transfermarkt
relabelled the absence table's duration column and moved the transfer history
behind a JSON endpoint, and both changes went unnoticed for two seasons: the
scrape caught the exceptions per player, so the only symptom was an
`absences_xxyy.csv` containing nothing but injuries. Every packaged file from
24/25 onwards is missing its suspensions and international call-ups as a result.

To refresh one, fetch the page and `gzip -9` it into place. Keep them gzipped -
the four together are 400KB of HTML and 60KB compressed.
