import json

# Rule-based data from collect_data.py
rule_easy = [0.99,0.88,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.88,0.99]
rule_medium = [0.789,0.789,0.689,0.889,0.286,0.889,0.689,0.183,0.889,0.689,0.186,0.286,0.286,0.286,0.286,0.889,0.689,0.689,0.689,0.889]
rule_hard = [0.29,0.19,0.19,0.19,0.29,0.29,0.19,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.788,0.29,0.19,0.784,0.29]

# LLM agent data (3 runs x 3 tasks)
llm_easy =   [0.800, 0.900, 0.001]
llm_medium = [0.001, 0.800, 0.800]
llm_hard =   [0.800, 0.001, 0.900]

def stats(data, name):
    avg = round(sum(data)/len(data), 3)
    variance = round(sum((x-avg)**2 for x in data)/len(data), 3)
    wins = sum(1 for x in data if x > 0.5)
    print(name + ':')
    print('  avg=' + str(avg) + ' variance=' + str(variance) + ' wins=' + str(wins) + '/' + str(len(data)))
    print('  min=' + str(min(data)) + ' max=' + str(max(data)))
    bimodal = sum(1 for x in data if x < 0.4) 
    print('  failure_episodes=' + str(bimodal))

print('=== RULE-BASED AGENT ===')
stats(rule_easy, 'easy')
stats(rule_medium, 'medium')
stats(rule_hard, 'hard')

print()
print('=== LLM AGENT (heuristic override active) ===')
stats(llm_easy, 'easy')
stats(llm_medium, 'medium')
stats(llm_hard, 'hard')

print()
print('=== KEY FINDING ===')
rb_avg = round((sum(rule_easy)+sum(rule_medium)+sum(rule_hard))/(len(rule_easy)+len(rule_medium)+len(rule_hard)),3)
llm_avg = round((sum(llm_easy)+sum(llm_medium)+sum(llm_hard))/(len(llm_easy)+len(llm_medium)+len(llm_hard)),3)
print('Rule-based overall avg: ' + str(rb_avg))
print('LLM overall avg: ' + str(llm_avg))
print()
print('Medium bimodal gap: ' + str(round(max(rule_medium)-min(rule_medium),3)))
print('Hard occasional breakthrough: ep16=0.788, ep19=0.784')
print('LLM never calls model - runs on pure heuristic override')
