## this season's players and fixtures
python fill_player_table.py
python fill_fixture_table.py
## historical matches and scores
python fill_match_table.py --input_type csv
python fill_playerscore_table.py
# fifa team ratings
python fill_fifa_ratings_table.py

## this season's results and scores

python fill_match_table.py --input_type api
python fill_playerscore_this_season.py

python fill_transaction_table.py

python fill_predictedscore_table.py --weeks_ahead 5
