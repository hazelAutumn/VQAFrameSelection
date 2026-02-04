import os.path as osp

#Process mylabel.txt file to filterout the lines with not existing video files
def filter_labels(label_file, video_dir, output_file):
    try:
        with open(label_file, "r") as fin:
            print(f"Processing label_file: {label_file}")
            for line in fin:
                line_split = line.strip().split(",")
                fileid, _, _, label = line_split
                label = float(label)
                filename = osp.join(video_dir, fileid)
                if osp.exists(filename):
                    with open(output_file, "a") as fout:
                        print(f"Processing line: {line.strip()}")
                        fout.write(line)
    except Exception as e:
        print(f"Error processing label_file: {e}")


def main():
    label_file = "./examplar_data_labels/YouTubeUGC/mylabels.txt"
    video_dir = "../../../VQA-datasets/InTheWild/Youtube-UGC/"
    output_file = "./examplar_data_labels/YouTubeUGC/mylabels_filtered.txt"
    filter_labels(label_file, video_dir, output_file)

if __name__ == "__main__":
    main()