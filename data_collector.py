import requests
import json


def get_text_from_url(url):
    r = requests.get(url)
    return r.text


def get_all_user_games(username):
    active_months_dict = json.loads(get_text_from_url(f"https://api.chess.com/pub/player/{username}/games/archives"))
    if 'archives' not in active_months_dict.keys():
        print(f"Error getting archive from chess.com API, {active_months_dict}")
        return
    active_months_urls = active_months_dict['archives']

    games = [monthly_games for active_month_url in active_months_urls for monthly_games in
             json.loads(get_text_from_url(active_month_url))['games']]
    return games
