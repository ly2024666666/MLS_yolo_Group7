import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo_format(xml_file, image_width, image_height):
    """
    Converts XML file to YOLO format
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_lines = []
    
    # Iterate through each object in the XML file
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_index = get_class_index(class_name)

        # Get the bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Calculate the center and size of the bounding box in YOLO format
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / float(image_width)
        height = (ymax - ymin) / float(image_height)

        # Create a line in the YOLO format: class_index x_center y_center width height
        yolo_line = f"{class_index} {x_center} {y_center} {width} {height}\n"
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def get_class_index(class_name):
    """
    Returns the class index based on class name.
    Define your classes here.
    """
    class_dict = {
        "smoking": 0,      # Class 0 for smoking
        "non-smoking": 1,  # Class 1 for non-smoking
    }
    return class_dict.get(class_name, 0)  # Default -1 if class is not found


def convert_all_xmls_to_yolo_format(xml_folder, image_folder, output_folder):
    """
    Converts all XML files in a folder to YOLO format TXT files.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each XML file in the folder
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            
            # Get image name without extension
            image_name = os.path.splitext(xml_file)[0]  # Remove '.xml' to get image base name
            
            # Check if the image file exists (in case paths are different)
            image_path = os.path.join(image_folder, image_name + ".jpg")  # Adjust this to your image format (e.g., .png)
            
            if os.path.exists(image_path):
                # Get image dimensions
                image_width, image_height = get_image_dimensions(image_path)
                
                # Convert the XML to YOLO format
                yolo_lines = convert_xml_to_yolo_format(xml_path, image_width, image_height)
                
                # Write the YOLO format to a text file
                txt_path = os.path.join(output_folder, image_name + ".txt")
                with open(txt_path, 'w') as f:
                    f.writelines(yolo_lines)
                print(f"Converted {xml_file} to {image_name}.txt")
            else:
                print(f"Image file not found for {xml_file}!")


def get_image_dimensions(image_path):
    """
    Returns the width and height of an image.
    Uses PIL to open the image and get its dimensions.
    """
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)


if __name__ == "__main__":
    xml_folder = r"C:\Users\alicerunqi\Downloads\pp_smoke\Annotations"  # Path to the folder containing XML files
    image_folder = r"C:\Users\alicerunqi\Downloads\pp_smoke\images"  # Path to the folder containing images
    output_folder = r"C:\Users\alicerunqi\Downloads\pp_smoke\txt_output_folder"  # Path where the TXT files will be saved
    
    convert_all_xmls_to_yolo_format(xml_folder, image_folder, output_folder)
