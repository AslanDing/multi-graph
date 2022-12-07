import numpy as np

path = '../exp/data_for_show/paperid_venueid_dict.npy'
data = np.load(path,allow_pickle=True).item()

list0 = []
list1 = []
list2 = []
for d in data.keys():
    if data[d] == '1192710900':
        list0.append(d)
    elif data[d] == '2756071452':
        list1.append(d)
    elif data[d] == '1121227772':
        list2.append(d)

node_select = {}
node_select['1192710900'] = list0
node_select['2756071452'] = list1
node_select['1121227772'] = list2

np.save("../exp/data_for_show/node_select.npy",node_select)

venue_count = {}
for d in data.keys():
    if data[d] in venue_count.keys():
        venue_count[data[d]]+=1
    else:
        venue_count[data[d]] = 1

venue_count = sorted(venue_count.items(),key=lambda k: k [1],reverse=True)
print(venue_count)

# ('1192710900', 5634), ('2756071452', 4129), ('1121227772', 3953)
