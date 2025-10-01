import requests

url = "http://127.0.0.1:8000/run"  # use 127.0.0.1 to avoid odd resolver issues

# Longer timeouts: (connect_timeout, read_timeout)
with open("A1.pdf", "rb") as f:
    r = requests.post(
        url,
        files={"file": f},
        data={"schema": "en_plus"},
        timeout=(5, 900)  # 5s connect, 900s (15min) read
    )

print(r.status_code)
print(r.text)  # use .text first to see any errors


# import requests
# url = "http://127.0.0.1:8000/run"
# r = requests.post(
#     url,
#     files={"link": (None, "https://arxiv.org/abs/2004.14303"),
#            "schema": (None, "en")},
#     timeout=(5, 120)  # arXiv should be fast
# )
# print(r.status_code)
# print(r.text)

