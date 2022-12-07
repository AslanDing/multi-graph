import os
import numpy as np
import json
import ijson

json_path = r'/media/aslan/50E4BE16E4BDFDF2/DATA/DATASET/dblp_papers_v11.txt'

paperid_name_dict={}
paperid_title_dict={}
paperid_authorid_dict={}
authorid_org_dict={}
authorid_name_dict={}
paperid_venue_dict={}
paperid_year_dict={}
paperid_refer_dict={}
paperid_fos_dict={}
with open(json_path,'r',encoding='utf-8') as fp:
    try:
        while True:
            line = fp.readline()
            if line:
                #print(line)
                #pass
                data = json.loads(line)
            else:
                break
    except Exception as e:
        print(e)
        fp.close()
    # data = json.load(fp)

    parser = ijson.items(fp)
    data =[]
    for prefix,event,value in parser:
        if event == "start_map":
            sin = {}
        print("event")
    # try:
    #     while True:
    #         line = fp.readline()
    #         if line:
    #             data = json.loads(line)
    #         else:
    #             break
    # except Exception as e:
    #     print(e)
    #     fp.close()
    #data = json.load(fp)
print("data")


""""
{ 
    "_id" : "53e99784b7602d9701f3e133", 
    "title" : "The relationship between canopy parameters and spectrum of winter wheat under different irrigations in Hebei Province.", 
    "authors" : [
        {
            "_id" : "53f45728dabfaec09f209538", 
            "name" : "Peijuan Wang"
        }, 
        {
            "_id" : "5601754345cedb3395e59457", 
            "name" : "Jiahua Zhang"
        }, 
        {
            "_id" : "53f38438dabfae4b34a08928", 
            "name" : "Donghui Xie"
        }, 
        {
            "_id" : "5601754345cedb3395e5945a", 
            "name" : "Yanyan Xu"
        }, 
        {
            "_id" : "53f43d25dabfaeecd6995149", 
            "name" : "Yun Xu"
        }
    ], 
    "venue" : {
        "_id" : "53a7297d20f7420be8bd4ae7", 
        "name_d" : "International Geoscience and Remote Sensing Symposium", 
        "type" : NumberInt(0), 
        "raw" : "IGARSS"
    }, 
    "year" : NumberInt(2011), 
    "keywords" : [
        "canopy parameters", 
        "canopy spectrum", 
        "different soil water content control", 
        "winter wheat", 
        "irrigation", 
        "hydrology", 
        "radiometry", 
        "moisture", 
        "indexes", 
        "vegetation", 
        "indexation", 
        "dry weight", 
        "soil moisture", 
        "water content", 
        "indexing terms", 
        "spectrum", 
        "natural disaster"
    ], 
    "fos" : [
        "Agronomy", 
        "Moisture", 
        "Hydrology", 
        "Environmental science", 
        "Dry weight", 
        "Water content", 
        "Stomatal conductance", 
        "Transpiration", 
        "Irrigation", 
        "Soil water", 
        "Canopy"
    ], 
    "n_citation" : NumberInt(0), 
    "page_start" : "1930", 
    "page_end" : "1933", 
    "lang" : "en", 
    "volume" : "null", 
    "issue" : "null", 
    "issn" : "", 
    "isbn" : "", 
    "doi" : "10.1109/IGARSS.2011.6049503", 
    "pdf" : null, 
    "url" : [
        "http://dx.doi.org/10.1109/IGARSS.2011.6049503"
    ], 
    "abstract" : "Drought is the first place in all the natural disasters in the world. It is especially serious in North China Plain. In this paper, different soil water content control levels at winter wheat growth stages are performed on Gucheng Ecological-Meteorological Integrated Observation Experiment Station of CAMS, China. Some canopy parameters, including growth conditions, dry weight, physiological parameters and hyperspectral reflectance, are measured from erecting stage to milk stage for winter wheat in 2009. The relationship between canopy parameters and soil relative moisture, canopy water content and water indices of winter wheat are established. The results show that some parameters, such as SPAD and dry weight of leaves, decrease with the increasing of soil relative moisture, while other parameters, including dry weight of caudexes, above ground dry weight, height, photosynthesis rate, intercellular CO 2 concentration, stomatal conductance and transpiration rate, increase corresponding to the soil relative moisture. Obvious linear relationship between stomatal conductance and transpiration rate is established with 45 samples, which R2 reaches to 0.6152. Finally, the fitting equations between canopy water content and water indices are regressed with b5, b6 and b7 of MODIS bands. The equations are best with b7 and worst with b5. So the fitting equations with b7 can be used to inverse the canopy water content of winter wheat using MODIS or other remote sensing images with similar bands range to MODIS in Hebei Province. © 2011 IEEE."
},
"""
