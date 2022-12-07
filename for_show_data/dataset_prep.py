import os
import numpy as np
import json
import ijson



json_path = r'/media/aslan/50E4BE16E4BDFDF2/DATA/DATASET/dblp_papers_v11.txt'
count = 0
paperid_title_dict={}
paperid_authorid_dict={}
paperid_venueid_dict={}
paperid_year_dict={}
paperid_refer_dict={}
paperid_fos_dict={}

venueid_name_dict={}

authorid_org_dict={}
authorid_name_dict={}
org_name=set()

with open(json_path,'r',encoding='utf-8') as fp:
    while count<500000:
        line = fp.readline()
        if line:
            data = json.loads(line)
            try:
                paper_id = data['id']
                paper_title = data['title']
                author_list = data['authors']
                paper_venue_name = data['venue']['raw']
                paper_venue_id = data['venue']['id']
                paper_year = data['year']
                paper_fos = data['fos']
                paper_references_list = data['references']

                author_id_list=[]
                for author in author_list:
                    author_id = author['id']
                    author_name = author['name']
                    author_org = author['org']
                    org_name.add(author_org)
                    authorid_org_dict[author_id]=author_org
                    authorid_name_dict[author_id]= author_name
                    author_id_list.append(author_id)

                paperid_title_dict[paper_id] = paper_title
                paperid_authorid_dict[paper_id] = author_id_list
                paperid_venueid_dict[paper_id] = paper_venue_id
                paperid_year_dict[paper_id] = paper_year
                paperid_refer_dict[paper_id] = paper_references_list
                paperid_fos_dict[paper_id] = paper_fos

                venueid_name_dict[paper_venue_id] = paper_venue_name
                count+=1
                print(count)
            except Exception as e:
                print(e)
                # fp.close()

        else:
            continue

print("finished")
np.save("../exp/data_for_show/paperid_title_dict.npy",paperid_title_dict)
np.save("../exp/data_for_show/paperid_authorid_dict.npy",paperid_authorid_dict)
np.save("../exp/data_for_show/paperid_venueid_dict.npy",paperid_venueid_dict)
np.save("../exp/data_for_show/paperid_year_dict.npy",paperid_year_dict)
np.save("../exp/data_for_show/paperid_refer_dict.npy",paperid_refer_dict)


np.save("../exp/data_for_show/venueid_name_dict.npy",venueid_name_dict)


np.save("../exp/data_for_show/authorid_org_dict.npy",authorid_org_dict)
np.save("../exp/data_for_show/authorid_name_dict.npy",authorid_name_dict)
np.save("../exp/data_for_show/org_name.npy",org_name)

# to feature
np.save("../exp/data_for_show/paperid_fos_dict.npy",paperid_fos_dict)

