from args import Args
from datasets import build_dataset
from model import load_model, run_inference
from bbox_filter import main_filter, class_logits_to_class
from generate_images import *
from generate_json import make_info_json
from transform_image import transform

import argparse
import torch
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and save results.")
    parser.add_argument('--data_nr', type=int, default=30, help='Index of the data sample to process')
    parser.add_argument('--filter_rate', type=float, default=4.5, help='Filter rate threshold')
    parser.add_argument('--file', type=str, help='Image file path')

    return parser.parse_args()

def main():
    start_time = time.time()

    args = parse_args()

    model_args = Args()
    device = torch.device(model_args.device)

    model = load_model(model_args, device)
    #dataset = build_dataset('val',model_args)

    #data_nr = args.data_nr
    #image_tensor = dataset[data_nr][0]
    
    t = transform(args.file)

    model.to('cpu')
    output = run_inference(t,model,device)#dataset[data_nr][0]

    filtered = main_filter(output, args.filter_rate)
    #pred_obj_logits = filtered['pred_obj_logits']
    pred_boxes = filtered['pred_obj_boxes'][0]

    labels = class_logits_to_class(filtered)
    show_image_boxes(t,pred_boxes,'pred',labels)
    #show_image_boxes_without_labels(t, pred_boxes, 'pred')
    make_info_json(filtered)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()