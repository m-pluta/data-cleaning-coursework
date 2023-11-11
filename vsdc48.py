import pandas as pd
import numpy as np
from collections import Counter
import re

OUT_FILEPATH = 'output_data.xlsx'
##### Problem 1

def initialCleaning(df):
    # Drop columns where all entries are NaN
    df.dropna(axis = 1, how="all", inplace=True)

    # Drop rows where all values are NaN
    df.dropna(how='all', inplace=True)

    # Remove all laptops that don't include any model numbers
    # Reasoning: Laptops with no model number will be very difficult for the person to find.
    df.dropna(subset=['model'], inplace=True)

    # Remove duplicate rows
    # Reasoning: Redundant information that unnecessarily skews the data
    df = df.drop_duplicates(ignore_index=True)

    # Convert all rows to lowercase
    # Reasoning: For consistency in data between columns, Different cases increase the dimensionality of the data.
    #            One case also allows easy comparisons between data and checking for duplicates.
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    df.reset_index(drop=True, inplace=True)

    return df

def extractDataFromModels(df):
    extractions = [r'(detachable 2-in-1|detachable 2 in 1)', r'(detachable)', r'(2 in 1|2-in-1)']

    for extraction in extractions:
        extracted_values = df['model'].str.extract(extraction, expand=False)
        df['special_features'] = df['special_features'].fillna('') + ', ' + extracted_values.fillna('')
        df['special_features'] = df['special_features'].str.strip(", ")
        df['model'] = df['model'].replace(extraction, '', regex=True)

    df['special_features'] = df['special_features'].str.replace('2 in 1', '2-in-1')
    return df

def normaliseModels(df):
    df['model'] = df['model'].str.replace('laptop', '')
    df = extractDataFromModels(df)
    return df

def standardiseColours(df):
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
        # r".*(information not available|rgb backlit|touchscreen|evo i7-1260p|acronym).*": np.NaN #TODO extract information from here
    }
    df['color'] = df['color'].replace(colorHashMap, regex=True)

    return df

def standardiseBrands(df):
    # Reasoning: Similar to standardisation of colours
    brandHashMap = {
        r".*(enovo).*": "lenovo",
        r".*(carlisle foodservice products|best notebooks|computer upgrade king|quality refurbished computers|microtella|ctl|lpt|rokc|elo|gizpro|jtd).*": np.NaN,
        # r".*(toughbook).*": "panasonic", #TODO move toughbook to the model later
        # r".*(latitude).*": "dell" #TODO ensure all latitude dells have dell as brand and latitude as model
    }
    df['brand'] = df['brand'].replace(brandHashMap, regex=True)

    return df

def extract_numerical_value(price):
    if pd.isna(price):
        return np.nan
    numerical_value = re.search(r'\d+(\.\d+)?', price.replace(",", "")).group()
    return float(numerical_value)

def cleanScreensPrices(df):
    # Remove 'inches' from screen sizes and '$' from prices. Convert these values to floats
    # Reasoning: Removing 'inches' and '$' allows the values to be converted from a categorical string type to a continuous scale.
    #            This allows for better data visualisation.
    df['screen_size'] = df['screen_size'].apply(extract_numerical_value)
    df['price'] = df['price'].apply(extract_numerical_value)
    return df

def convert_to_gb(size, tb_possible = False):
    # TODO check the correctness of this function, especially with NaNs
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

def convertRamDiskSizesToGB(df):
    # Adjust harddisk sizes to be more consistent
    df['harddisk'] = df['harddisk'].apply(convert_to_gb, tb_possible=True)
    df['ram'] = df['ram'].apply(convert_to_gb, tb_possible=False)

    return df

def bucketStorageSize(size: int) -> int:
    if pd.isna(size):
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
def bucketHardDisk(df):
    df['harddisk'] = df['harddisk'].apply(bucketStorageSize)

    return df

def standardiseOS(df):
    # Convert os systems to standard systems
    osHashMap = {
        r".*(windows 10 pro|win 10 pro).*": "windows 10 pro",
        r".*(windows 11 pro).*": "windows 11 pro",
        r".*(windows 10|win 10).*": "windows 10 home",
        r".*(windows 11 home|win 11 multi-home).*": "windows 11 home",
        r".*(windows 7 professional).*": "windows 7 pro"
    }
    df['OS'] = df['OS'].replace(osHashMap, regex=True)

    return df

def normaliseCPUSpeeds(df):
    # Adjust CPU speeds to be more consistent with the same units
    def normaliseCPU(cpu_speed):
        cpu_speed = str(cpu_speed)
        match = re.search(r'((\d+\.\d+|\d+))', cpu_speed)
        if match:
            value = float(match.group())
            if value > 10:
                value /= 1000
            return value
  
    df['cpu_speed'] = df['cpu_speed'].apply(normaliseCPU)

    return df

def renameColumns(df):
    columnNameHashMap = {
        'screen_size': 'screen_size_inches',
        'color': 'colour',
        'harddisk': 'harddisk_gb',
        'ram': 'ram_gb',
        'cpu_speed': 'cpu_speed_ghz',
        'price': 'price_usd'
    }
    return df.rename(columns=columnNameHashMap)

def outputDataToFile(df):
    df.to_excel(OUT_FILEPATH, index=False)  # index=False to exclude the index column in the output file

if __name__ == "__main__":
    # function_calls = [initialCleaning, normaliseModels, standardiseColours, standardiseBrands, 
    #                  cleanScreensPrices, convertRamDiskSizesToGB, standardiseOS, 
    #                  normaliseCPUSpeeds, renameColumns]

    function_calls = [initialCleaning, normaliseModels]
    
    df = pd.read_excel('amazon_laptop_2023.xlsx')
    for function in function_calls:
        df = function(df)
    outputDataToFile(df)

    ### TESTING ###

    print("Total rows: ", df.shape[0])
    # print(df.dtypes)


    # unique_elems = Counter(df['model'].dropna().tolist())
    # for el in unique_elems:
    #     print(el)

    # Most common words in the model column
    myCounter = Counter(' '.join(df['model'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if not word.isdigit()]))).split())
    for el, count in myCounter.most_common():
        print(f"{el}: {count}")
