import os
import pandas as pd

# Folder containing TXT files
text_folder = r"C:\Users\Manoj\OneDrive\Desktop\dm assignment\dataset\dutch_text"

# Path to truth file
truth_file = r"C:\Users\Manoj\OneDrive\Desktop\dm assignment\dataset\dutch_text\truth.txt"

# Store labels
labels = {}

# Read truth file
with open(truth_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")
        
        userid = parts[0]
        extroversion_score = float(parts[3])

        # Convert score to label
        label = "extrovert" if extroversion_score >= 0.5 else "introvert"

        labels[userid] = label

# Store dataset rows
data = []

# Read TXT files
for filename in os.listdir(text_folder):

    if filename.endswith(".txt"):

        userid = filename.replace(".txt", "")

        txt_path = os.path.join(text_folder, filename)

        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Get label from truth file
        label = labels.get(userid, "unknown")

        # Add row
        data.append([userid, text, label])

# Create DataFrame
df = pd.DataFrame(data, columns=["userid", "text", "label"])

# Save as CSV
output_path = r"C:\Users\Manoj\OneDrive\Desktop\dm assignment\dataset\dutch_dataset.csv"

df.to_csv(output_path, index=False)

print("CSV dataset created successfully!")