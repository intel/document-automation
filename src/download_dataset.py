from dataset_librarian.dataset_api.download import download_dataset
from dataset_librarian.dataset_api.preprocess import preprocess_dataset

def parse_cmd():
    desc = 'download DuReader-vis dataset for preprocessing...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    
     # directories
    args.add_argument('--dataset_dir', type=str, default='/home/user/dataset/', dest='dataset_dir')
 
    print(args.parse_args())
    return args.parse_args()

if __name__ == "__main__":
    
    cfg = parse_cmd()

    # Download the datasets
    download_dataset('dureader-vis', cfg.dataset_dir)