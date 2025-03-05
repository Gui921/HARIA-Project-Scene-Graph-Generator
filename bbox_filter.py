def filter_keys(data, list_index):
    
    filtered_data = {}

    for key, value in data.items():
        if key == 'aux_outputs':
            continue

        filtered_data[key] = value[:,list_index]

    return filtered_data

def main_filter(output, filter_rate):
    obj_logits = output['pred_obj_logits'][0]
    filtered_obj_indexes = []

    for i in range(len(obj_logits)):
        pred = obj_logits[i]
        pred_lst = pred.tolist()[:-1]

        if max(pred_lst) > filter_rate:
            filtered_obj_indexes.append(i)

    filtered_obj = filter_keys(output, filtered_obj_indexes)
    return filtered_obj

def class_logits_to_class(filtered):
    nr_bboxes = len(filtered['pred_obj_boxes'][0])
    objects = []
    for i in range(nr_bboxes):
        values_lst = filtered['pred_obj_logits'][0][i].tolist()
        max_obj_value = max(values_lst[:-1])
        max_obj_index = values_lst.index(max_obj_value)
        objects.append(max_obj_index)
    
    return objects