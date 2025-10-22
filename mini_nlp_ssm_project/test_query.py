from model import most_similar
query = 'I Love Python'

result = most_similar(query)
for sent,score in result:
    print(sent,"-->",score)