"""Send a city-day to the running prediction service and print the answer.

    uv run python scripts/test_service.py                                  # the local service, demo.json
    uv run python scripts/test_service.py --url https://<deployed-service> # a deployed one
    uv run python scripts/test_service.py --dangerous-threshold 0.1        # with an alert threshold
    uv run python scripts/test_service.py --live Delhi                     # today's forecast from live data

Start the service first:  uv run uvicorn predict:app --app-dir scripts --port 8080
"""
import argparse
import json
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--url', default='http://localhost:8080', help='where the service runs')
    parser.add_argument('--input', default=PROJECT_DIR / 'demo.json', help='JSON file with one city-day')
    parser.add_argument('--live', metavar='CITY', help="today's forecast for this city from live data (live=true)")
    parser.add_argument('--severe-threshold', type=float)
    parser.add_argument('--dangerous-threshold', type=float)
    args = parser.parse_args()
    url = args.url.rstrip('/')

    health = requests.get(f'{url}/health', timeout=90)
    health.raise_for_status()
    print(f"GET  {url}/health -> {health.json()['status']}, {len(health.json()['cities'])} cities")

    params = {k: v for k, v in [('severe_threshold', args.severe_threshold),
                                ('dangerous_threshold', args.dangerous_threshold)] if v is not None}
    if args.live:
        record, params['live'] = {'city': args.live}, 'true'
    else:
        record = json.loads(Path(args.input).read_text())
    # A live request downloads and reads the bulletins the first time after the service starts: allow time
    response = requests.post(f'{url}/predict', json=record, params=params, timeout=300 if args.live else 90)
    print(f'POST {url}/predict{"?live=true" if args.live else ""} ({record["city"]}) -> HTTP {response.status_code}')
    print(json.dumps(response.json(), indent=2))
    response.raise_for_status()


if __name__ == '__main__':
    main()
