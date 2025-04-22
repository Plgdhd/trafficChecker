import pandas as pd

data = [
    ("GET /index.html HTTP/1.1 Host:example.com", "normal"),
    ("POST /login.php HTTP/1.1 Host:example.com\nusername=admin&password=123", "normal"),
    ("GET /../../etc/passwd HTTP/1.1", "malicious"),
    ("POST /upload.php HTTP/1.1\npayload=<script>alert(1)</script>", "malicious"),
    ("GET /dashboard HTTP/1.1 Host:example.com", "normal"),
    ("POST /config HTTP/1.1\nrm -rf /", "malicious"),
    ("GET /api/data?id=5 HTTP/1.1", "normal"),
    ("GET /?q=<img src=x onerror=alert(1)> HTTP/1.1", "malicious"),
]

df = pd.DataFrame(data, columns=["request_data", "label"])
df.to_csv("http_requests.csv", index=False)
print("CSV создан");