import requests
import time
print("Waiting 3 minutes...")
time.sleep(180)
base = 'https://Fieerawe-cybersec-soc-env.hf.space'
for ep in ['/failure_analysis','/simulator','/red_team_reasoning','/ciso_report','/alert_fatigue']:
    r = requests.get(base+ep, timeout=10)
    print(ep, 'PASS' if r.status_code==200 else f'FAIL {r.status_code}')