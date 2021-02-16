#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

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

