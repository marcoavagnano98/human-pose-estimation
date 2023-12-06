import os 
import shutil

TRAIN_FOLDER = "dataset/technogym_ds/train"
TEST_FOLDER = "dataset/technogym_ds/test"
TEST_SET = ["20231201_095755", 
            "20231201_095817", 
            "20231201_095839",
            "20231201_100048",
            "20231201_100537"
            ]

RGB_FOLDER = "rgb_unaligned"

tr_count, ts_count = 0, 0
# Create TEST and TRAIN SET
for root_dir, cur_dir, files in os.walk("frames/lontano"):
    if RGB_FOLDER in root_dir and not "background" in root_dir:
        _,_,bag_name, _ =  root_dir.split("/")
        if not bag_name in TEST_SET:
            print(f"Add {bag_name} to training set")
            for file in files:
                fpath = os.path.join(root_dir, file)
                destpath = os.path.join(TRAIN_FOLDER, f"{tr_count:06d}.jpg")
                shutil.copyfile(fpath, destpath)
                tr_count += 1
        else:
            print(f"Add {bag_name} to test set")
            for file in files:
                fpath = os.path.join(root_dir, file)
                destpath = os.path.join(TEST_FOLDER, f"{ts_count:06d}.jpg")
                shutil.copyfile(fpath, destpath)
                ts_count += 1