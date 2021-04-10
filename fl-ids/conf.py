from datetime import datetime

RUN_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
RUN_TIME = datetime.now().strftime(RUN_TIME_FORMAT)

LABEL_TO_ID = {  
    "Benign": 0,
    'Infilteration': 1,
    'DoS attacks-Slowloris': 2,
    'SSH-Bruteforce': 3,
    'DDOS attack-HOIC': 4,
    'FTP-BruteForce': 5,
    'DoS attacks-SlowHTTPTest': 6,
    'Bot': 7,
    'DoS attacks-Hulk': 8,
    'DoS attacks-GoldenEye': 9,
    'DDoS attacks-LOIC-HTTP': 10,
    'DDOS attack-LOIC-UDP': 11,
    'Brute Force -Web': 12,
    'Brute Force -XSS': 13,
    'SQL Injection': 14,
}

ID_TO_LABEL = {
    v: k for k, v in LABEL_TO_ID.items()
}