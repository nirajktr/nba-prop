from nba_api.stats.static import teams

def main():
    nba_teams = teams.get_teams()
    print("nba_api imported successfully.")
    print(f"Total teams retrieved: {len(nba_teams)}")
    print(f"First team in list: {nba_teams[0]['full_name']}")

if __name__ == "__main__":
    main()