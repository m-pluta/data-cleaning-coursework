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

def normaliseModels(df):
    # Reasoning: A laptop being a 2-in-1 or being detachable is more of a special feature. 
    #            Hence it should be extracted from the model column and placed in the special_features column
    def extractDetachable2in1(row):
        regexHashMap = {
            r'(detachable 2-in-1|detachable 2 in 1)': 'detachable 2-in-1',
            r'(detachable)': 'detachable',
            r'(2 in 1|2-in-1)': '2-in-1'
        }
        for pattern in regexHashMap:
            if re.search(pattern, str(row['model'])):
                target = 'special_features'
                row[target] = regexHashMap[pattern] if pd.isna(row[target]) else (row[target] + ', ' + regexHashMap[pattern])
                row['model'] = re.sub(pattern, '', str(row['model']))

        return row
    df = df.apply(extractDetachable2in1, axis=1)

    # Reasoning: Only 6 laptops contain the release year so there isn't enough data and also the release year isnt important enough info
    regexModelYear = r'(\(2021\)|\(2022\)|2021|2022|2023)'
    df['model'] = df['model'].str.replace(regexModelYear, '', regex=True)

    # Reasoning: The screen size is already included in the screen_size column and it is often more precise in that column
    #            This was verified by examining the data, therefore redundant data
    regexScreenSize = r'((\d{2}(?:\.\d{1,2})?)[ -]?(?:inch|\"))'
    df['model'] = df['model'].str.replace(regexScreenSize, '', regex=True)

    # Reasoning: No need to keep model data within brackets, clearer without
    #            Only affects a small portion of total rows, determined by examining the data
    df['model'] = df['model'].str.replace(r'\(|\)', '', regex=True)

    # Reasoning: Marketing gibberish is not useful
    #            Pointless information, the dataset is about laptops
    df['model'] = df['model'].str.replace(r'newest|laptop', '', regex=True)

    # Reasoning: Only use spaces as seperation characters for consistency
    df['model'] = df['model'].str.replace(r'[(?:\s+),/\\\-_\*]', ' ', regex=True)


    # Extract DD.D screen sizes

    # Extract D.D CPU speeds

    # Extract w7p, w10p Check if the existing OS value matches

    # Extract CPU brands and series

    # Remove the 'victus by hp'

    # Extract 1yr warranty to special features

    # Extract ram and storage values and compare with existing





    

    # Reasoning: Remove unnecessary characters from the column
    df['model'] = df['model'].replace('\s+', ' ', regex=True)
    df['model'] = df['model'].str.strip(', ')
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

def printNumEmpty(df):
    emptyStrings = df.eq('').sum()
    nulls = df.isnull().sum()

    print('COLUMN_NAME'.ljust(21) + '\'\''.rjust(3) + 'null'.rjust(7))

    for column in df.columns.tolist():
        print(column.ljust(21), end="")
        print(str(emptyStrings[column]).rjust(3), end="")
        print(str(nulls[column]).rjust(7), end="\n")

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

    # print("Total rows: ", df.shape[0])
    # print(df.dtypes)
    printNumEmpty(df)

    # # Most common words in the model column
    # myCounter = Counter(' '.join(df['model'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if not word.isdigit()]))).split())
    # for el, count in myCounter.most_common():
    #     print(f"{el}: {count}")
