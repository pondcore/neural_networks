import requests, time
from numpy import random
t=0
vote = 0
vote_num = 5
payload = []
r = []
url = 'https://gz9ofg2pnc.execute-api.ap-southeast-1.amazonaws.com/Prod/api/items/TSM20200304/vote'
payload.append({'iid': vote_num, 'uid': 'c6d7f831-71a1-46e6-9e64-2ed8665f0dc5', 'react': 'vote'})
# payload1 = {'iid': vote_num, 'uid': '10fb5ac9-3dcc-4b19-b8bf-e5df4548492c', 'react': 'vote'}
# payload2 = {'iid': vote_num, 'uid': '83f1cdf4-05bb-4468-bb42-060649446f1a', 'react': 'vote'}
# payload3 = {'iid': vote_num, 'uid': '0cf024b4-96a5-41a5-8e14-9dee92279864', 'react': 'vote'}
# payload4 = {'iid': vote_num, 'uid': '7866a8bd-a91f-41c4-81e6-200d9b04b164', 'react': 'vote'}
r.append(requests.post(url, data=payload[0]))
# r1 = requests.post(url, data=payload1)
# r2 = requests.post(url, data=payload2)
# r3 = requests.post(url, data=payload3)
# r4 = requests.post(url, data=payload4)
for i in range(len(payload)):
    print(r[i].json()['list'])
pass

while(True):
    t+=1
    time.sleep(1)
    for i in range(len(payload)):
        if time.time()*1000 > int(r[i].json()['list']['nextVoteTime']) :
            time.sleep(random.randint(3))
            r[i] = requests.post(url, data=payload[i])
            vote += 1
            print(r[i].json()['list'])
        pass
    print(vote)
    if t%2 == 0 :
        print('/')
        if t == 10:
            t = 0
    else:
        print('\\')