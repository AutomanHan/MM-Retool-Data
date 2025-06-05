import os
import base64

import requests

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    with open(
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/linminliang/.vscode/gemini.key"
    ) as fp:
        _GEMINI_API_KEY = fp.read().strip()


def bin_to_base64(path):
    with open(path, "rb") as fp:
        bin_data = fp.read()
    base64_data = base64.b64encode(bin_data)
    return base64_data.decode("utf-8")


def call_gemini(model, messages):
    def compose_answer(response):
        res = []
        for x in response:
            if "content" in x["candidates"][0]:
                res.append(x["candidates"][0]["content"]["parts"][0]["text"])
        return "".join(res)

    response = requests.post(
        f"https://aigc.sankuai.com/v1/google/models/{model}:streamGenerateContent",
        headers={
            "Authorization": f"Bearer {_GEMINI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=messages,
    )
    response.raise_for_status()
    data = response.json()
    return compose_answer(data)