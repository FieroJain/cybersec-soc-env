import json

rule_easy = [0.99,0.88,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.88,0.99]
rule_medium = [0.789,0.789,0.689,0.889,0.286,0.889,0.689,0.183,0.889,0.689,0.186,0.286,0.286,0.286,0.286,0.889,0.689,0.689,0.689,0.889]
rule_hard = [0.29,0.19,0.19,0.19,0.29,0.29,0.19,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.788,0.29,0.19,0.784,0.29]

topology_data = {
    'hierarchical': [0.286,0.286,0.883,0.789,0.186,0.286,0.889,0.689,0.286],
    'mesh':         [0.28,0.889,0.889,0.689,0.689,0.789,0.889],
    'segmented':    [0.186,0.186,0.286],
    'star':         [0.783,0.789,0.789,0.789,0.689,0.186,0.889,0.689,0.28,0.186,0.689]
}

print('=== FINAL RESEARCH SUMMARY ===')
print()
print('EXPERIMENT 1: Rule-based agent across difficulty levels (n=60)')
for name, data in [('easy', rule_easy), ('medium', rule_medium), ('hard', rule_hard)]:
    avg = round(sum(data)/len(data), 3)
    wins = sum(1 for x in data if x > 0.5)
    failures = sum(1 for x in data if x < 0.4)
    variance = round(sum((x-avg)**2 for x in data)/len(data), 4)
    print(name + ': avg=' + str(avg) + ' wins=' + str(wins) + '/20 failures=' + str(failures) + '/20 variance=' + str(variance))

print()
print('EXPERIMENT 2: Topology vs containment rate (n=30, medium task)')
for topo, scores in sorted(topology_data.items()):
    avg = round(sum(scores)/len(scores), 3)
    wins = sum(1 for s in scores if s > 0.5)
    failures = sum(1 for s in scores if s < 0.4)
    win_rate = round(wins/len(scores)*100)
    print(topo + ': avg=' + str(avg) + ' win_rate=' + str(win_rate) + '% n=' + str(len(scores)))

print()
print('TOPOLOGY RANKING (best to worst):')
avgs = [(t, round(sum(s)/len(s),3)) for t,s in topology_data.items()]
avgs.sort(key=lambda x: x[1], reverse=True)
for t, a in avgs:
    print('  ' + t + ': ' + str(a))

mesh_avg = sum(topology_data['mesh'])/len(topology_data['mesh'])
seg_avg = sum(topology_data['segmented'])/len(topology_data['segmented'])
print()
print('Mesh vs Segmented performance gap: ' + str(round(mesh_avg/seg_avg, 2)) + 'x')
