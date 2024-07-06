import pandas as pd
import os
import re

# Directory containing the CSV files
input_folder = 'discord-chat'
output_file = 'output.txt'

# Initialize a list to store the formatted messages
formatted_messages = []

# Regular expression pattern to match the custom emoji format
emoji_pattern = re.compile(r'<:[a-zA-Z0-9_]+:[0-9]+>')

# Loop through each CSV file in the directory
for i in range(1, 18):
    # Construct the file name
    file_name = f'chat{i}.csv'
    file_path = os.path.join(input_folder, file_name)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Remove the 'Date' and 'User tag' columns
    df = df.drop(columns=['Date', 'User tag'])
    
    # Format each message and add it to the list
    for _, row in df.iterrows():
        content = row['Content']
        # Check if content is a string, otherwise set it to an empty string
        if not isinstance(content, str):
            content = ''
        
        # Remove custom emojis from the content
        content_cleaned = re.sub(emoji_pattern, '', content)
        
        if pd.notna(row['Mentions']):
            message = f"{row['Username']} said {content_cleaned} to {row['Mentions']}"
        else:
            message = f"{row['Username']} said {content_cleaned}"
        
        formatted_messages.append(message)

# Write the formatted messages to the output file with UTF-8 encoding
with open(output_file, 'w', encoding='utf-8') as f:
    for message in formatted_messages:
        f.write(message + '\n')

print(f"Formatted messages saved to {output_file}")
