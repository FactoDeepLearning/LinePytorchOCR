import os
import shutil
import xml.etree.ElementTree as ET


xml_files_path = "./xml"  # Folder containing train.xml, valid.xml and test.xml
img_path = "./img"  # Content of IAM lines.tgz

output_path = "."  # where to store preformatted dataset
set_names = ["train", "valid", "test"]

output_folder = os.path.join(output_path, "../Datasets/iam_lines")
os.makedirs(output_path, exist_ok=True)
for set_name in set_names:
    id = 0
    current_folder = os.path.join(output_folder, set_name)
    os.makedirs(current_folder, exist_ok=True)
    xml_path = os.path.join(xml_files_path, "{}.xml".format(set_name))
    xml_root = ET.parse(xml_path).getroot()
    label_file_path = os.path.join(output_folder, "{}.txt".format(set_name))
    with open(label_file_path, "w") as f_out:
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(img_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                path_for_label = os.path.join(set_name, "{}.png".format(id))
                new_path = os.path.join(current_folder, "{}.png".format(id))
                shutil.copy(img_paths[i], new_path)
                f_out.write("{} {}\n".format(path_for_label, label))
                id += 1

