import json
import os

from class_and_labels import *

def make_info_json(filtered):
    nr_bboxes = len(filtered['pred_obj_boxes'][0])
    info = {}
    for i in range(nr_bboxes):
        object = {}
        values_lst = filtered['pred_obj_logits'][0][i].tolist()
        max_obj_value = max(values_lst[:-1])
        max_obj_index = values_lst.index(max_obj_value)

        spatial_values = filtered['pred_spatial_logits'][0][i].tolist()
        max_spatial_value = max(spatial_values[:-1])
        max_spatial_index = spatial_values.index(max_spatial_value)

        object['Spatial'] = SPATIAL_LABELS[max_spatial_index]

        contact_values = filtered['pred_contacting_logits'][0][i].tolist()
        max_contact_value = max(contact_values[:-1])
        max_contact_index = contact_values.index(max_contact_value)

        object['Contact'] = CONTACTING_LABELS_2[max_contact_index]

        attention_values = filtered['pred_attn_logits'][0][i].tolist()
        max_attention_value = max(attention_values[:-1])
        max_attention_index = attention_values.index(max_attention_value)

        object['Attention'] = ATTENTION_LABELS[max_attention_index]
        info[OBJ_CLASSES[max_obj_index]] = object
    
    json_object = json.dumps(info,indent=4)

    json_path = os.path.join('output', 'output')

    with open(json_path, 'w') as json_file:
        json_file.write(json_object)

    print(f"JSON file saved to {json_path}")
    #return json_object