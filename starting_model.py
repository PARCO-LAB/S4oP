import os
import argparse

import torch

from config import *
from efficient_pruning.model import ModelTrain, ModelTest
from efficient_pruning.model.utils import set_benchmark, set_seed

set_seed(42)
set_benchmark(False)


def main(model_name, dataset_name, epochs, batch_size, valsplit, checkpoints_folder, image_side):
    config = PRUNING_DEFAULT_CONFIG

    model_path = os.path.join(checkpoints_folder, "{}_{}_best.pth".format(model_name, dataset_name))
    lr = 0.001 if dataset_name == "imagenet" else 0.1
    milestones = [80, 100] if dataset_name == "imagenet" else [120, 150, 180]

    if not os.path.exists(model_path):
        model_train = ModelTrain.from_scratch(
            model_name, dataset_name, 
            batch_size=batch_size, valsplit=valsplit, num_workers=config["num_workers"], image_side=image_side)
        print(model_name, dataset_name)
        print(model_train.model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_train.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 150, 180], gamma=0.1)
        model_train.run(optimizer, criterion, epochs, scheduler=scheduler, checkpoints_folder=checkpoints_folder)
    else: 
        print("Skipping training because model path {} already exists".format(model_path))

    
    model_test = ModelTest.from_pth(
        model_path, 
        batch_size=config["test_batch_size"], 
        num_workers=config["num_workers"], 
        image_side=image_side)
    model_test.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "--model", "-m", 
        dest="model_name", 
        required=True,
        help="Model name")
    parser.add_argument(
        "--dataset", "-d", 
        dest="dataset_name", 
        required=True,
        help="Dataset name")
    parser.add_argument(
        "--epochs", "-e", 
        dest="epochs", 
        required=True,
        help="Epochs amount")
    parser.add_argument(
        "--batch-size", "-b", 
        dest="batch_size", 
        required=True,
        help="Batch size")
    parser.add_argument(
        "--valsplit", "-v", 
        dest="valsplit", 
        required=False, default=0.2,
        help="Val split percentage: [0, 1]")
    parser.add_argument(
        "--checkpoints-folder", "-f", 
        dest="checkpoints_folder", 
        required=False, default=os.path.join(".", "checkpoints"),
        help="Checkpoints folder")
    parser.add_argument(
        "--image-side",
        dest="image_side", 
        required=False, default=None,
        help="Checkpoints folder")
    args = parser.parse_args()
    main(args.model_name, args.dataset_name, 
         int(args.epochs), int(args.batch_size), float(args.valsplit), args.checkpoints_folder, 
         int(args.image_side) if args.image_side is not None else None)
