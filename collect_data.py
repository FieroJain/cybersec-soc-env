import json, requests, time

url = 'https://Fieerawe-cybersec-soc-env.hf.space'
results = {}

for task in ['easy', 'medium', 'hard']:
    print('Running ' + task + ' x 20 episodes...')
    results[task] = []
    for ep in range(20):
        r = requests.post(url + '/tasks/' + task + '/grade', params={'n_episodes': 1})
        d = r.json()
        results[task].append(d)
        score = d['score']
        containment = d['details']['containment_rate']
        print('  ep ' + str(ep+1) + ': score=' + str(score) + ' containment=' + str(containment))
        time.sleep(1)

with open('rule_based_20ep.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Done!')
