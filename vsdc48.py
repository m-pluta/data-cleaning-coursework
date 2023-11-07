import pandas as pd
import numpy as np
from collections import Counter
import re

##### Problem 1

df = pd.read_excel('amazon_laptop_2023.xlsx')

# Drop rows where all values are NaN
df.dropna(how='all', inplace=True)

# Convert all rows to lowercase
# Reasoning: For consistency in data between columns, Different cases increase the dimensionality of the data.
#            One case also allows easy comparisons between data and checking for duplicates.
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

# Remove duplicate rows
# Reasoning: Redundant information that unnecessarily skews the data
df = df.drop_duplicates(ignore_index=True)

# Remove all laptops that don't include any model numbers
# Reasoning: Laptops with no model number will be very difficult for the person to find. 
#            Many different manufacturers can create
df.dropna(subset=['model'], inplace=True)

# Convert colours to standard colours
# Reasoning: Reduces variability in the dataset due to misspellings, variations and interpretations
#            Reducuing the variability also helps with categorization which is use for data exploration and visualization
#            A part of this process also ensures contextual standardization which is crucial where colours are named uniquely 
#            for marketing or specific contexts (e.g. thunder black or platinum titan)
colorHashMap = {
    r".*(silver|sliver|aluminum|platinum|light titan|platinum titan).*": "silver",
    r".*(black|dark side of the moon|thunder balck|carbon fiber).*": "black",
    r".*(white).*": "white",
    r".*(grey|gray|gary|graphite|mercury|dark ash|dark metallic moon).*": "grey",
    r".*(red).*": "red",
    r".*(blue|cobalt|sky|dark teal|apollo|midnight).*": "blue",
    r".*(green|sage|soft mint|dark moss).*": "green",
    r".*(gold).*": "gold",
    r".*(almond|beige mousse|lunar light|dune).*": "beige", #TODO Need to justify this
    r".*(punk pink|electro punk).*": "pink",
    r".*(information not available|rgb backlit|touchscreen|evo i7-1260p|acronym).*": np.NaN #TODO extract information from here
}
df['color'] = df['color'].replace(colorHashMap, regex=True)

# Convert brands to standard brands
# Reasoning: Similar to standardisation of colours
brandHashMap = {
    r".*(enovo).*": "lenovo",
    r".*(carlisle foodservice products|best notebooks|computer upgrade king|quality refurbished computers|microtella|ctl|lpt|rokc|elo|gizpro|jtd).*": np.NaN,
    r".*(toughbook).*": "panasonic", #TODO move toughbook to the model later
    r".*(latitude).*": "dell" #TODO ensure all latitude dells have dell as brand and latitude as model
}
df['brand'] = df['brand'].replace(brandHashMap, regex=True)

# Remove 'inches' from screen sizes and convert screen sizes to float
# Reasoning: Removing inches allows the values to be converted from a categorical to a continuous scale.
#            This allows for better data visualisation attempts.
df['screen_size'] = df['screen_size'].str.replace(' inches', '').astype(float)


# Adjust harddisk sizes to be more consistent
# TODO check the correctness of this function, especially with NaNs
def convert_to_gb(size, tb_possible:bool = False):
    size = str(size)

    if pd.isna(size): # Check for NaN
        return size
    
    if size.endswith('tb'):
        return round(float(size.replace(' tb', '')) * 1000)
    
    if size.endswith('gb'):
        return round(float(size.replace(' gb', '')))
    
    elif size.endswith('mb'):
        return round(float(size.replace(' mb', '')))
    else:
        if tb_possible and float(size) < 16:
            return float(size) * 1000
        else: 
            return float(size)       
df['harddisk'] = df['harddisk'].apply(convert_to_gb, tb_possible=True)
df['ram'] = df['ram'].apply(convert_to_gb, tb_possible=False)


def check_unique_sizes(size: int) -> int:
    if pd.isna(size): # Check for NaN
        return size
    
    if size in (64, 65):
        return 64
    if size in (120, 128):
        return 128
    if size in (240, 250, 256):
        return 256
    if size in (480, 500, 512):
        return 512
    if size in (1000, 1024):
        return 1024
    if size in (2000, 2048):
        return 2048
    if size in (4000, 4096):
        return 4096
    if size in (8000, 8192):
        return 8192

    return size
# df['harddisk'] = df['harddisk'].apply(check_unique_sizes)




# Convert os systems to standard systems
osHashMap = {
    r".*(windows 10 pro|win 10 pro).*": "windows 10 pro",
    r".*(windows 11 pro).*": "windows 11 pro",
    r".*(windows 10|win 10).*": "windows 10 home",
    r".*(windows 11 home|win 11 multi-home).*": "windows 11 home",
    r".*(windows 7 professional).*": "windows 7 pro"
}
df['OS'] = df['OS'].replace(osHashMap, regex=True)


# Adjust CPU speeds to be more consistent with the same units
def normaliseCPU(cpu_speed):
    cpu_speed = str(cpu_speed)
    match = re.search(r'((\d+\.\d+|\d+))', cpu_speed)
    if match:
        value = float(match.group())
        if value > 6:
            value /= 1000
df['cpu_speed'] = df['cpu_speed'].apply(normaliseCPU)



# Testing using watch
print("Total rows: ", df.shape[0])

unique_elems = Counter(df['model'].dropna().tolist())
for el in unique_elems:
    print(el)

# Most common words in the model column
# print(Counter(' '.join(df['model'].astype(str).apply(lambda x: ' '.join(word for word in x.split() if word.isalpha())).fillna('')).split()))



def outputDataToFile():
    output_file_name = 'output_data.xlsx'

    df.to_excel(output_file_name, index=False)  # index=False to exclude the index column in the output file
outputDataToFile()