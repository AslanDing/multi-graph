import numpy as np

# author org author
author_path = '../exp/data_for_show/select_authorid_dict.npy'
author_name_path = '../exp/data_for_show/authorid_name_dict.npy'
author_org_path = '../exp/data_for_show/authorid_org_dict.npy'
author_list = np.load(author_path,allow_pickle=True).item()
author_org = np.load(author_org_path,allow_pickle=True).item()
author_name_dict = np.load(author_name_path,allow_pickle=True).item()

# author_org_author = np.zeros([23794,23794])

count = 0
author_dict = {}
author_name = []
author_orgs = []
org_count = 0
org_dict = {}
for author in author_list.keys():
    if author in author_dict.keys():
        continue
    else:
        author_dict[author] = count
        author_name.append(author_name_dict[author])
        count+=1

    org = author_org[author]
    author_orgs.append(org)
    if org in org_dict.keys():
        # org_dict[org] += 1
        continue
    else:
        org_dict[org] = org_count
        org_count += 1
print()
np.save("../exp/data_for_show/select_author_name.npy",author_name)
np.save("../exp/data_for_show/select_author_org.npy",author_orgs)

# org_dict = sorted(org_dict.items(), key=lambda k: k[1], reverse=True)

author_org_m=np.zeros([count,org_count])
for author in author_list.keys():
    org = author_org[author]
    author_org_m[author_dict[author],org_dict[org]] = 1

# print()
author_org_author = author_org_m @ author_org_m.T
np.save("../exp/data_for_show/select_author_org_author.npy",author_org_author)


#label
path = '../exp/data_for_show/node_select.npy'
selet_list = np.load(path,allow_pickle=True).item()

title_path = '../exp/data_for_show/paperid_title_dict.npy'
title = np.load(title_path,allow_pickle=True).item()

venueid_path = '../exp/data_for_show/paperid_venueid_dict.npy'
venueid = np.load(venueid_path,allow_pickle=True).item()

venuename_path = '../exp/data_for_show/venueid_name_dict.npy'
venuename = np.load(venuename_path,allow_pickle=True).item()

year_path = '../exp/data_for_show/paperid_year_dict.npy'
year = np.load(year_path,allow_pickle=True).item()

paperid_dict = {}
count = 0

paper_title = []
paper_avenue = []
paper_year = []

for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        if node_id in paperid_dict.keys():
            continue
        else:
            paperid_dict[node_id] = count
            count+=1
        paper_title.append(title[node_id])
        paper_avenue.append(venuename[venueid[node_id]])
        paper_year.append(year[node_id])

print()
np.save("../exp/data_for_show/select_paper_name.npy",paper_title)
np.save("../exp/data_for_show/select_paper_venue.npy",paper_avenue)
np.save("../exp/data_for_show/select_paper_year.npy",paper_year)



label = []
count = 0
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        label.append(count)
    count += 1
np.save("../exp/data_for_show/select_paper_label.npy",label)



train_list=[]  # 0.7
test_list=[]
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    train_count = int(0.7 * len(list_select))
    for node_id in range(0,train_count):
        train_list.append(paperid_dict[list_select[node_id]])
    for node_id in range(train_count,len(list_select)):
        test_list.append(paperid_dict[list_select[node_id]])
split_dict = {"train":train_list,"test":test_list}
np.save("../exp/data_for_show/select_paper_split.npy",split_dict)


# author fetaures
paper_features_path = r"../exp/data_for_show/select_node_feature.npy"
paper_features = np.load(paper_features_path,allow_pickle=True)
paper_authors_path = "../exp/data_for_show/select_paperid_authorid.npy"
paper_authors = np.load(paper_authors_path,allow_pickle=True)
print()
author_features = paper_authors.T @ paper_features
np.save("../exp/data_for_show/select_author_feature.npy",author_features)




# authors
paperid_author_path = '../exp/data_for_show/paperid_authorid_dict.npy'
authorid = np.load(paperid_author_path,allow_pickle=True).item()
count = 0
author_ids = {}
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        authors = authorid[node_id]
        for au in authors:
            if au in author_ids.keys():
                continue
            else:
                author_ids[au] = count
                count += 1

        #print()
count = 0
paperid_authorid = np.zeros([13716,23794])
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        authors = authorid[node_id]
        for au in authors:
            paperid_authorid[count,author_ids[au]] = 1
        count += 1

np.save("../exp/data_for_show/select_paperid_authorid.npy",paperid_authorid)
np.save("../exp/data_for_show/select_authorid_dict.npy",author_ids)

# paper cite paper
paper_preference_path = '../exp/data_for_show/paperid_refer_dict.npy'
paperid_cite_paperid = np.zeros([13716,13716])
reference_list = np.load(paper_preference_path,allow_pickle=True).item()



for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        reference = reference_list[node_id]
        # print()
        for refer in reference:
            if refer in paperid_dict.keys():
                paperid_cite_paperid[paperid_dict[refer],paperid_dict[node_id]] = 1

print()
np.save("../exp/data_for_show/paperid_cite_paperid.npy",paperid_cite_paperid)


# feature
nodes_features = '../exp/data_for_show/paperid_fos_dict.npy'

nodes_f = np.load(nodes_features,allow_pickle=True).item()

print("test")

# 40

feature_names = {}
feature_names_set = set()
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        feature = nodes_f[node_id]
        for i in feature:
            feature_names_set.add(i['name'])
            if i['name'] in feature_names.keys():
                feature_names[i['name']] += 1
            else:
                feature_names[i['name']] = 1

feature_names = sorted(feature_names.items(),key=lambda k: k [1],reverse=True)
#print(feature_names_set)

count =0
feature_name_dict = {}
for name in feature_names:
    if count<40:
        feature_name_dict[name[0]] = count
        count+=1
    else:
        break

count = 0
features=np.zeros([13716,40])
for select_l in selet_list.keys():
    list_select = selet_list[select_l]
    for node_id in list_select:
        feature = nodes_f[node_id]
        for i in feature:
            if i['name'] in feature_name_dict.keys():
                features[count,feature_name_dict[i['name']]] = i['w']
        count+=1

print('feature')
np.save("../exp/data_for_show/select_node_feature.npy",features)

