import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_excel('amazon_laptop_2023.xlsx')

# Drop rows where all values are NaN
df.dropna(how='all', inplace=True)



# Convert all rows to lowercase
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

# Remove duplicate rows
df = df.drop_duplicates()




# Convert colours to standard colours
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
    r".*(information not available|rgb backlit|touchscreen|evo i7-1260p|acronym).*": np.NaN
}
df['color'] = df['color'].replace(colorHashMap, regex=True)




# Convert brands to standard brands
brandHashMap = {
    r".*(enovo).*": "lenovo",
    r".*(carlisle foodservice products|best notebooks|computer upgrade king|quality refurbished computers|microtella|ctl|lpt|rokc|elo|gizpro|jtd).*": np.NaN,
    r".*(toughbook).*": "panasonic", #TODO move toughbook to the model later
    r".*(latitude).*": "dell"
}
df['brand'] = df['brand'].replace(brandHashMap, regex=True)




# Remove 'inches' from screen sizes and convert screen sizes to float
df['screen_size'] = df['screen_size'].str.replace(' inches', '').astype(float)




# Adjust harddisk sizes to be more consistent

def convert_to_gb(harddisk):
    size = str(harddisk)

    if pd.isnull(size):
        return size
    elif size.endswith('tb'):
        return float(size.replace(' tb', '')) * 1000
    elif size.endswith('gb'):
        return float(size.replace(' gb', ''))
    elif size.endswith('mb'):
        return float(size.replace(' mb', ''))
    else:
        if float(size) < 16:
            return float(size) * 1000
        else:
            return float(size)
    
    return np.NaN

df['harddisk'] = df['harddisk'].apply(convert_to_gb)


# Output data to file
def outputDataToFile():
    output_file_name = 'output_data.xlsx'

    df.to_excel(output_file_name, index=False)  # index=False to exclude the index column in the output file



# Testing using watch
print("Total rows: ", df.shape[0])

unique_elems = Counter(df['harddisk'].dropna().tolist())
print(unique_elems)

# print(Counter(' '.join(df['model'].astype(str).apply(lambda x: ' '.join(word for word in x.split() if word.isalpha())).fillna('')).split()))

outputDataToFile()