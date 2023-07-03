
import argparse
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from MSCNet import MSCNet
from utils.eval import EvalCocoDataset,evaluate_PFNet


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evalute PFNet on COCO dataset.')
    parser.add_argument('--dataset', required=False,
                        default="/CV/3T/dataset-public/COCO/coco2017",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default='2017',
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        default='/home/gaobiaoli/mapping/PFNet-pytorch/logs/pfnet_0100.pth',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--limit', required=False,
                        default=5000,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=5000), as mentioned in the paper')
    args = parser.parse_args()
    device = torch.device("cuda:0")
    random_rho=True
    mask=True
    # Create model
    model = MSCNet()

    # Load weights
    model_path = args.model
    print("Loading weights ", model_path)
    #torch.distributed.init_process_group('nccl',rank=dist.get_rank(),world_size=dist.get_world_size())
    pretrained_dict = torch.load(model_path,map_location=torch.device("cuda"))
    try:
        model.load_state_dict(pretrained_dict.module.state_dict())
    except:
        model.load_state_dict(pretrained_dict)
    model = model.to(device)
    model.eval()

    # Validation dataset
    dataset_val = EvalCocoDataset(args.dataset, "test", year=args.year,random_rho=random_rho)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    #summary(model, (2, 128, 128))
    print("Running COCO evaluation on {} images regarding PFNet performance.".format(args.limit))
    evaluate_PFNet(model, val_loader, int(args.limit), device,
                   normalize_factor=32,mask_mode=2)
    # rho = 4
    # end=36
    # while True:
    #     val_loader.dataset.rho = rho
    #     filename="mace_"+str(rho)+".npy"
    #     evaluate_PFNet(model, val_loader, int(args.limit), device, macePath=filename)
    #     rho+=4
    #     if rho>end:
    #         break
