import requests, json
r = requests.get('http://localhost:8000/api/reports/diag-exercises/FotKtoYUQ1gZVeddXE0hU1ZMotk1')
print("STATUS:", r.status_code)
print(json.dumps(r.json(), indent=2))
