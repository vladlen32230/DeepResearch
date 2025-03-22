import requests

import json

response = requests.get(

  url="https://openrouter.ai/api/v1/auth/key",

  headers={

    "Authorization": f"Bearer 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"

  }

)

print(json.dumps(response.json(), indent=2))