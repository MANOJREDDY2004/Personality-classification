import os
import xml.etree.ElementTree as ET

# Folder containing XML files
xml_folder = r"C:\Users\Manoj\OneDrive\Desktop\dm assignment\dataset\english"

# Folder to save TXT files
output_folder = r"C:\Users\Manoj\OneDrive\Desktop\dm assignment\dataset\english_text"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Read all XML files
for filename in os.listdir(xml_folder):

    if filename.endswith(".xml"):

        xml_path = os.path.join(xml_folder, filename)

        # Parse XML file
        tree = ET.parse(xml_path)

        root = tree.getroot()

        # Store all text/posts
        texts = []

        # Extract documents/posts
        for document in root.iter("document"):

            if document.text:
                texts.append(document.text.strip())

        # Combine all posts into one text
        full_text = " ".join(texts)

        # Create TXT filename
        txt_filename = filename.replace(".xml", ".txt")

        txt_path = os.path.join(output_folder, txt_filename)

        # Save as TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Converted: {filename} -> {txt_filename}")

print("\nAll XML files converted successfully!")