import requests, json, time

url = 'https://Fieerawe-cybersec-soc-env.hf.space'
topology_results = []

print('Collecting topology vs outcome data...')
for i in range(30):
    r = requests.post(url + '/reset')
    obs = r.json()['observation']
    topology = obs['topology_type']
    
    r2 = requests.post(url + '/tasks/medium/grade', params={'n_episodes': 1})
    d = r2.json()
    
    topology_results.append({
        'topology': topology,
        'score': d['score'],
        'containment': d['details']['containment_rate'],
        'false_isolations': d['details']['false_isolations']
    })
    print('run ' + str(i+1) + ': topology=' + topology + ' score=' + str(d['score']) + ' containment=' + str(d['details']['containment_rate']))
    time.sleep(1)

with open('topology_analysis.json', 'w') as f:
    json.dump(topology_results, f, indent=2)

by_topology = {}
for r in topology_results:
    t = r['topology']
    if t not in by_topology:
        by_topology[t] = []
    by_topology[t].append(r['score'])

print()
print('=== SCORE BY TOPOLOGY ===')
for t, scores in sorted(by_topology.items()):
    avg = round(sum(scores)/len(scores), 3)
    wins = sum(1 for s in scores if s > 0.5)
    print(t + ': avg=' + str(avg) + ' wins=' + str(wins) + '/' + str(len(scores)) + ' scores=' + str(scores))
