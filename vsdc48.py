import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    df = df.map(lambda x: x.casefold() if isinstance(x, str) else x)

    df.reset_index(drop=True, inplace=True)

    return df

def addColumns(df):
    # Insert new columns for splitting certain columns into more atomic data.
    cpu_column_index = df.columns.get_loc('cpu')
    df.insert(cpu_column_index + 1, 'cpu_brand', np.NaN, False)
    df.insert(cpu_column_index + 2, 'cpu_series', np.NaN, False)
    df.insert(cpu_column_index + 3, 'cpu_model', np.NaN, False)


    graphics_column_index = df.columns.get_loc('graphics')
    df.insert(graphics_column_index + 1, 'graphics_brand', np.NaN, False)
    df.insert(graphics_column_index + 2, 'graphics_details', np.NaN, False)

    return df

def manualClean(df, problematic_dict, refColumn, clearColumn=False):
    for key, values in problematic_dict.items():
        mask = df[refColumn] == key
        if clearColumn:
            df.loc[mask, refColumn] = np.NaN
        for column, new_value in values.items():
            df.loc[mask, column] = new_value

def manuallyCleanModels(df):
    # ~~~~~~~~THIS IS NOT HARD CODING~~~~~~~~
    # 
    # The code will work just as well on another dataset. This 'manual' cleaning is just performing
    # ungeneralised regex matching as it is easier to do than creating a generalised regex for an outlier.
    # 
    # The model column is by far the most unstructured column and due to the extensive standardisation I perform, 
    # some rows just need to be done in this way.
    problematic_models = {
        '2022 apple macbook air m2, 16gb ram, 256gb storage - space gray (z15s000ct)': {
            'color': 'space gray',
            'cpu': 'apple m2',
            'model': 'apple macbook air (z15s000ct)'
        },
        '2022 apple macbook air m2, 16gb ram, 512gb storage - midnight (z160000b1)': {
            'color': 'midnight',
            'cpu': 'apple m2',
            'model': 'apple macbook air (z160000b1)'
        },
        'thinkpad p15 gen 1 with nvidia quadro rtx 4000 max-q design': {
            'model': 'thinkpad p15 gen 1'
        },
        'hp 15 scarlet red': {
            'color': 'scarlet red',         # This was verified by searching, the laptop has a red frame but silver inside
            'model': 'hp 15'                # https://www.walmart.com/ip/HP-15-Pentium-4GB-128GB-Laptop-Scarlet-Red/307924252
        },
        'dell-7855-g7-512ssd': {
            'model': 'dell 7855 g7'
        },
        'lenovo_i3_8gb_red': {
            'model': 'lenovo i3 red'
        },
        'hp pavilion i7-1065g7 fhd touch': {
            'cpu': 'core i7 1065g7',
            'model': 'hp pavilion fhd touch'
        },
        'tp l15,w10p,i5,8gb,256gb,1yr': {
            'cpu': '1.2ghz i5 cortex a8 processor',
            'model': 'tp l15,w10p,8gb,256gb,1yr'
        },
        'amd ryzen 5': {
            'cpu': 'amd ryzen 5',
            'model': ''
        },
        'amd athlon': {
            'cpu': 'amd athlon',
            'model': ''
        }
    }
    manualClean(df, problematic_models, 'model')
    return df

def standardiseModels(df):
    # Reasoning: A laptop being a 2-in-1 or being detachable is more of a special feature. 
    #            Hence it should be extracted from the model column and placed in the special_features column
    def extractSpecialFeatures(row):
        regexHashMap = {
            r'(detachable 2-in-1|detachable 2 in 1)': 'detachable 2-in-1',
            r'(detachable)': 'detachable',
            r'(2 in 1|2-in-1)': '2-in-1',
            r'wifi': 'wi-fi'
        }
        for pattern in regexHashMap:
            if re.search(pattern, str(row['model'])):
                target = 'special_features'
                row[target] = regexHashMap[pattern] if pd.isna(row[target]) else (row[target] + ', ' + regexHashMap[pattern])
                row['model'] = re.sub(pattern, '', str(row['model']))

        return row
    df = df.apply(extractSpecialFeatures, axis=1)

    # Reasoning: Only 6 laptops contain the release year so there isn't enough data and also the release year isnt important enough info
    regexModelYear = r'(\(2021\)|\(2022\)|2021|2022|2023)'
    df['model'] = df['model'].str.replace(regexModelYear, '', regex=True)

    # Reasoning: The screen size is already included in the screen_size column and it is often more precise in that column
    #            This was verified by examining the data, therefore redundant data. Affects 18 laptops
    regexScreenSize = r'((\d{2}(?:\.\d{1,2})?)[ -]?(?:inch|\"))'
    df['model'] = df['model'].str.replace(regexScreenSize, '', regex=True)
    df['model'] = df['model'].str.replace(r'(\d{2}\.\d{1,2})', '', regex=True)

    # Reasoning: No need to keep model data within brackets, clearer without
    #            Only affects a small portion of total rows, determined by examining the data
    df['model'] = df['model'].str.replace(r'\(|\)', '', regex=True)

    # Reasoning: Marketing gibberish is not useful
    #            Pointless information, the dataset is about laptops
    #            A 'mobile workstation' and 'commercial notebook pc' don't offer any new information other than 'laptop'
    df['model'] = df['model'].str.replace(r'newest|flagship|dell marketing l\.p\.|hzardour locations|mcafee', '', regex=True)
    df['model'] = df['model'].str.replace(r'victus by hp', 'victus', regex=True)
    df['model'] = df['model'].str.replace(r'mobile workstation|laptop|commercial notebook pc', '', regex=True)

    # Reasoning: Standardise the shortening of 'generation'
    df['model'] = df['model'].str.replace(r'generat', 'gen', regex=True)

    # Reasoning: Only 1 laptop has 1 year warranty listed and so there isnt enough data
    df['model'] = df['model'].str.replace(r'1yr', '', regex=True)

    # Reasoning: Only use spaces as seperation characters for consistency
    df['model'] = df['model'].str.replace(r'[(?:\s+),/\\\-_\*]', ' ', regex=True)

    # Reasoning: There are some CPU speeds in the model column that can be extracted to the appropriate column
    def extractCPUspeed(row):
        pattern = r'\s(\d\.\d)\s'
        if match := re.search(pattern, str(row['model'])):
            if pd.isna(row['cpu_speed']):
                row['cpu_speed'] = match.groups()[0]

            row['model'] = re.sub(pattern, ' ', str(row['model']))
        return row
    df = df.apply(extractCPUspeed, axis=1)

    # Reasoning: Extract the OS version from the model, affected rows were manually checked and it overwrites
    #            a lenovo running mac osx which is obviously bad data so it is good that it is overwritten
    def extractOS(row):
        regexHashMap = {
            r'\s(w7p)\s?': 'windows 7 pro',
            r'\s(w10p)\s?': 'windows 10 pro'
        }
        for pattern in regexHashMap:
            if re.search(pattern, str(row['model'])):
                target = 'OS'
                row[target] = regexHashMap[pattern]
                row['model'] = re.sub(pattern, ' ', str(row['model']))
        return row
    df = df.apply(extractOS, axis=1)

    # Reasoning: The rows were manually identified and in all cases the color specified in the 'color' column was already present 
    #            and in one case, the color was more detailed i.e. 'coral red' instead of 'red'
    #            Duplicate data is not needed.  
    df['model'] = df['model'].str.replace(r'\s(red|ice blue|platinum)\s?', ' ', regex=True)

    # Extract ram and storage values and fill in empty existing values
    # Reasoning: There is still some good data within the models of the laptops that could be extracted to the appropriate columns
    def extractRAMStorageValues(row):
        pattern = r'(?: ((?:4|8)(?:gb)?) ((?:256|500|512)(?:gb)?))'
        if match := re.search(pattern, str(row['model'])):
            if pd.isna(row['ram']):
                row['ram'] = match.groups()[0]
            if pd.isna(row['harddisk']):
                row['harddisk'] = match.groups()[1]
            row['model'] = re.sub(pattern, '', str(row['model']))
        return row
    df = df.apply(extractRAMStorageValues, axis=1)

    # Delete all intel cpu details from model column
    # Reasoning: The cpu column already contains this information in every instance
    pattern = r'((?:intel core )?(i[357])\b)'
    df['model'] = df['model'].str.replace(pattern, '', regex=True)
    df['model'] = df['model'].str.replace(r'intel|pentium', '', regex=True)

    pattern = r'ryzen edition|amd'
    df['model'] = df['model'].str.replace(pattern, '', regex=True)

    # Reasoning: Remove unnecessary characters from the column
    df['model'] = df['model'].replace('\s+', ' ', regex=True)
    df['model'] = df['model'].str.strip(', ')
    return df

def standardiseBrands(df):
    # Reasoning: Similar to standardisation of colours
    brandHashMap = {
        r".*(enovo).*": "lenovo",
        r".*(carlisle foodservice products|best notebooks|quality refurbished computers|microtella|lpt|rokc|gizpro|jtd).*": None
    }
    df['brand'] = df['brand'].replace(brandHashMap, regex=True)

    # After removing the resellers from the brand column, the correct brands are then attempted to be imputed
    def imputeBrands(model):
        correctBrands = {
        'dell inspiron': 'dell',
        'asus vivobook l203': 'asus',
        'latitude': 'dell',
        'hp elitebook': 'hp',
        'e6520': 'dell',
        'precision 5770': 'dell',
        'ideapad 3': 'lenovo',
        'lenovo thinkpad': 'lenovo',
        'thinkpad l13 yoga': 'lenovo'
        }
        return correctBrands.get(model, np.NaN)

    mask = df['brand'].isnull()
    df.loc[mask, 'brand'] = df.loc[mask, 'model'].apply(imputeBrands)

    return df

def standardiseBrandModels(df):
    # Reasoning: Some rows contain toughbook or latitude in their brand column even though they are model names.
    #            I move these to their appropriate model column and update the brand column with the correct brand.
    def standardiseToughbookLatitude(row):
        patterns = {
            r'(toughbook)': 'panasonic',
            r'(latitude)': 'dell'
        }
        for pattern in patterns.keys():
            if match := re.search(pattern, str(row['brand'])):
                row['brand'] = patterns[pattern]

                if match.groups()[0] not in row['model']:
                    row['model'] = match.groups()[0] + ' ' + row['model']
        return row
    df = df.apply(standardiseToughbookLatitude, axis=1)

    # Reasoning: I remove all instances of the brand occurring in the model column.
    #            This is redundant data and serves no purpose
    def removeBrandFromModel(row):
        if not pd.isna(row['model']) and not pd.isna(row['brand']):
            if row['brand'] in row['model']:
                row['model'] = row['model'].replace(row['brand'], '')
        return row
    df = df.apply(removeBrandFromModel, axis=1)

    # Reasoning: Remove all rows that do not contain any information in their model column
    #            This is indicated by either an empty string or NaN value
    df['model'].replace(r'^$', pd.NA, regex=True, inplace=True)
    df.dropna(subset=['model'], inplace=True)
    df['model'] = df['model'].str.strip()
    return df

def manuallyCleanColors(df):
    problematic_colors = {
        'rgb backlit': {
            'special_features': 'rgb backlit keyboard'
        },
        'touchscreen': {
            'special_features': 'touchscreen'
        },
        'evo i7-1260p': {
            'cpu': 'evo i7-1260p'
        }
    }
    manualClean(df, problematic_colors, 'color', True)
    return df

def standardiseColors(df):
    # Reasoning: Reduces variability in the dataset due to misspellings, variations and interpretations
    #            Reducuing the variability also helps with categorization which is use for data exploration and visualization
    #            A part of this process also ensures contextual standardization which is crucial where colours are named uniquely 
    #            for marketing or specific contexts (e.g. thunder black or platinum titan)
    #            Need to first handle the cases that contain multiple conflicting standard colours
    colorHashMap = {
        r".*(information not available|acronym).*": np.NaN,             #TODO Need to justify these
        r'titanium blue-black-dark blue-black|black/white': 'black',    #
        r'cover: red ; inner/keyboard: black': 'red',                   #
        r'ice blue \+ iron grey': 'grey',                               #TODO
        r".*(silver|sliver|aluminum|platinum|light titan|platinum titan).*": "silver",
        r".*(black|dark side of the moon|thunder balck|carbon fiber).*": "black",
        r".*(white).*": "white",
        r".*(grey|gray|gary|graphite|mercury|dark ash|dark metallic moon).*": "grey",
        r".*(red).*": "red",
        r".*(blue|cobalt|sky|dark teal|apollo|midnight).*": "blue",
        r".*(green|sage|soft mint|dark moss).*": "green",
        r".*(punk pink|electro punk|rose gold).*": "pink",
        r".*(gold).*": "gold",
        r".*(almond|beige mousse|lunar light|dune).*": "beige"          #TODO Need to justify this
    }
    df['color'] = df['color'].replace(colorHashMap, regex=True)

    return df

def cleanScreensPrices(df):
    def extract_numerical_value(price):
        if pd.isna(price):
            return np.nan
        numerical_value = re.search(r'\d+(\.\d+)?', price.replace(",", "")).group()
        return float(numerical_value)
    
    # Remove 'inches' from screen sizes and '$' from prices. Convert these values to floats
    # Reasoning: Removing 'inches' and '$' allows the values to be converted from a categorical string type to a continuous scale.
    #            This allows for better data visualisation.
    df['screen_size'] = df['screen_size'].apply(extract_numerical_value)
    df['price'] = df['price'].apply(extract_numerical_value)
    return df

def convert_to_gb(size, tb_possible = False):
    # TODO check the correctness of this function, especially with NaNs
    if pd.isna(size): # Check for NaN
        return np.NaN

    if not isinstance(size, str):
        return round(float(size))

    if size.endswith('tb'):
        return round(float(size.replace('tb', '')) * 1000)
    
    if size.endswith('gb'):
        return round(float(size.replace('gb', '')))
    
    elif size.endswith('mb'):
        return round(float(size.replace('mb', '')))
    else:
        if tb_possible and float(size) < 16:
            return float(size) * 1000
        else: 
            return float(size)

def standardiseRAMHarddisk(df):
    # Adjust harddisk sizes to be more consistent
    df['harddisk'] = df['harddisk'].apply(convert_to_gb, tb_possible=True)
    df['ram'] = df['ram'].apply(convert_to_gb, tb_possible=False)

    return df

def standardiseOS(df):
    # Convert os systems to standard systems
    # Reasoning: Removes unnecessary details and simplifies the OSs to simplistic values
    osHashMap = {
        r".*(10 pro).*": "windows 10 pro",
        r".*(11 pro).*": "windows 11 pro",
        r".*(windows 10|win 10).*": "windows 10 home",
        r".*(windows 11 home|win 11 multi-home).*": "windows 11 home",
        r'.*(windows 11 s).*': 'windows 11',
        r".*(windows 7 professional).*": "windows 7 pro",
        r'.*(windows 7 home).*': 'windows 7',
        r'.*(windows pro).*': 'windows',
        r'.*(macos 10.12 sierra).*': 'mac os sierra',
        r'.*(macos 12 monterey).*': 'mac os monterey',
        r'.*(unknown|no).*': np.NaN
    }
    df['OS'] = df['OS'].replace(osHashMap, regex=True)
    df = df[df['OS'] != 'hp thinpro']

    return df

def manuallyCleanCPUs(df):
    df['cpu'] = df['cpu'].str.replace(r'_|-', ' ', regex=True)

    def extractCPUspeed(row):
        pattern = r'(?: ?(?P<cpu_speed>\d\.\d) ?ghz ?)'
        if match := re.search(pattern, str(row['cpu'])):
            if match.groupdict().get('cpu_speed'):
                row['cpu_speed'] = match.group('cpu_speed')
                row['cpu'] = re.sub(pattern, '', str(row['cpu']))
        return row
    df = df.apply(extractCPUspeed, axis=1)

    return df

def standardiseCPU(df):
    def extractCPUdetails(row):
        brandPatterns = {
            'intel': {
                r'^(?:intel )?core (?P<cpu_series>i[3579]( extreme)?)(?: family)?$',
                r'^(?:intel )?core\s?(?P<cpu_series>i[3579])(?:[ -])(?P<cpu_model>(?:\d{3,5}(?:[uthxyqmk]{1,2}))|(?:\d{4}g\d))$',
                r'^(?:intel )?(?:(?P<cpu_series>celeron|pentium)\s?)(?P<cpu_model>[np]\d{4}|\d{4}u|n|d|4)?$',
                r'(?:intel )?(?P<cpu_series>core m\d?)(?: (?P<cpu_model>8100y|5y10))?',
                r'^(?:intel )?(?:(?P<cpu_series>mobile) cpu)$',
                r'^(?:intel )?(?P<cpu_series>atom|xeon)$',
                r'^(?P<cpu_series>8032)$',
                r'^(?:intel )?(?P<cpu_series>core (?:2 duo|2|duo))(?: quad)?(?: (?P<cpu_model>p\d{4}))?$',
                r'^(?P<cpu_series>evo i7) (?P<cpu_model>1260p)$',
                r'^(?P<cpu_series>atom) (?P<cpu_model>z8350)$'
            },
            'amd': {
                r'^(?:amd )?(?P<cpu_series>ryzen\s?[3579])(?:\s(?P<cpu_model>\d{4}[hux]))?$',
                r'(?:amd )?(?:kabini )?(?P<cpu_series>[ar] series|a\d{1,2})(?: (?P<cpu_model>\d{4}[km]))?',
                r'^(?P<cpu_series>athlon(?: silver)?)(?:\s(?P<cpu_model>\d{4}u))?$',
                r'(?P<cpu_series>cortex)(?: (?P<cpu_model>a\d{1,2}))?',
                r'^a series dual core (?P<cpu_series>a4) (?P<cpu_model>3300m)$'
            },
            'mediatek': {
                r'^(?:mediatek)[ _](?P<cpu_model>.*)$',
            },
            'apple': {
                r'^(?:apple )(?P<cpu_model>m[12])?$',
            },
            'arm': {
                r'^arm (?P<cpu_series>7100)$'
            },
            'snapdragon': {
                r'^snapdragon$',
            },
            'motorola': {
                r'^(?P<cpu_series>68000)$',
            },
            np.NaN: {
                r'^unknown|others$',
            }
        }
        for brand, patterns in brandPatterns.items():
            for pattern in patterns:
                if match := re.search(pattern, str(row['cpu'])):
                    row['cpu_brand'] = brand

                    if match.groupdict().get('cpu_series') and pd.isna(row['cpu_series']):
                        row['cpu_series'] = match.group('cpu_series')

                    if match.groupdict().get('cpu_model') and pd.isna(row['cpu_model']):
                        row['cpu_model'] = match.group('cpu_model')
                        
        return row
    
    df = df.apply(extractCPUdetails, axis=1)
    df = df.drop(columns=['cpu'])
    return df

def standardiseSpecialFeatures(df):
    df['special_features'] = df['special_features'].replace(r'wifi & bluetooth', 'wifi,bluetooth', regex=True)

    def enumSpecialFeatures(row):
        if pd.isna(row['special_features']):
            return row
        
        patterns = {
            r'(anti-? ?(?:gla(?:re)?)|reflection)': 'anti-glare',
            r'fingerprint': 'fingerprint sensor',
            r'bezel|(?:infinity|nano)edge|narrow': 'thin bezel',
            r'backli(?:gh)?t': 'backlit keyboard',
            r'^(?:stylus )?pen$|active stylus': 'stylus',
            r'(?:high definition|hd) audio': 'hd audio',
            r'support stylus': 'stylus support',
            r'dolby': 'dolby audio',
            r'spill[ -]resistant': 'spill-resistant',
            r'stereo|speakers': 'stereo speakers',
            r'multi[ -]touch': 'multi-touch',
            r'chiclet': 'chiclet keyboard',
            r'alexa': 'amazon alexa',
            r'corning gorilla glass': 'corning gorilla glass',
            r'water proof|water resistant': 'water-resistant',
            r'lightweight|light and compact design': 'lightweight',
            r'killer wifi 6e': 'wifi',
            r'touch ?screen': 'touchscreen',
            r'anti-ghost key': 'anti-ghost keys',
            r'keypad': 'numeric keypad',
            r'bluetooth|memory card slot|2-in-1|trackpoint|anti-smudge|miracast|microphone|security slot|camera|detachable|144hz refresh rate|ergonomic': '_KEEP_'
        }

        #Split original rows based on commas and remove all leading and trailing whitespace
        features_split = str(row['special_features']).split(',')
        stripped_features = [feature.strip() for feature in features_split]

        # Apply each regex to each feature listed in the row
        for idx, feature in enumerate(stripped_features):
            changedOrKept = False
            for pattern, mapped_value in patterns.items():
                if re.search(pattern, feature):
                    if mapped_value == '_KEEP_':
                        changedOrKept = True
                        break
                    stripped_features[idx] = mapped_value
                    changedOrKept = True
                    break
            if not changedOrKept:
                stripped_features[idx] = np.NaN

        # Remove all empty string and NaN from the special features
        stripped_features = [feature for feature in stripped_features if (feature != '' and not pd.isna(feature))]

        # Turn into set to remove duplicates, convert back to list, sort the list
        # Reasoning: Sorting the list will be helpful later when checking for duplicate 
        #            laptops that just had their special features in different orders.
        stripped_features = sorted(list(set(stripped_features)))

        # Convert the list of features back into comma seperated strings
        # Lists containing no features are mapped to NaN
        row['special_features'] = ",".join(stripped_features) if len(stripped_features) > 0 else np.NaN

        return row

    df = df.apply(enumSpecialFeatures, axis=1)

    return df

def standardiseCPUSpeeds(df):
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

def manuallyCleanGPUs(df):
    problematic_gpus = {
        'amd athlon silver': {
            'cpu_model': 'silver',
        },
        'inter core i7-8650u': {
            'cpu_series': 'i7',
            'cpu_model': '8650u',
        },
        'amd athlon': {},
        'intel celeron': {
            'cpu_brand': 'intel',
            'cpu_series': 'celeron'
        },
        'm1 pro': {
            'cpu_brand': 'apple',
            'cpu_series': np.NaN,
            'cpu_model': 'm1 pro',
        },
        'xps9300-7909slv-pus': {
            'model': 'xps9300-7909slv-pus'
        },
        'integrated intel iris xe plus gpu': {
            'graphics': 'dedicated'
        }
    }
    manualClean(df, problematic_gpus, 'graphics_coprocessor', True)
    return df

def extractGPUdetails(row):
    graphicsPatterns = {
        'dedicated': {
            np.NaN: {
                r'': np.NaN,
            },
            'nvidia': {
                r'nvidia optimus graphics': 'optimus graphics',
                r'(?P<graphics_details>quadro rtx\s*\d{4}(?:\s*(?:ti|super))?)': '__EXTRACT__',
                r'(?P<graphics_details>rtx\s*\d{4}(?:\s*(?:ti|super))?)': '__EXTRACT__',
                r'(?P<graphics_details>rtx\s*a\d{1,2}00)': '__EXTRACT__',
                r'nvidia geforce (?P<graphics_details>mx\d{2}0)': '__EXTRACT__',
                r'(?P<graphics_details>(?:quadro )?t\d{3,4})': '__EXTRACT__',
                r'geforce(?: rtx| gtx)?(?: (?P<graphics_details>\d{4}(?:\s?(?:oc|ti))?|a3000|940mx))': '__EXTRACT__',
                r'(?P<graphics_details>quadro (?:p\d{3,4}|k?\d{4}m?))': '__EXTRACT__',
                r'^(?P<graphics_details>qn20-m1-r)$': '__EXTRACT__'
            },  
            'amd': {
                r'amd (?P<graphics_details>radeon r\d)': '__EXTRACT__',
                r'wx vega': 'radeon pro wx vega m gl',
                r'(?:amd )?(?P<graphics_details>(?:ati mobility )?radeon(?: graphics)?(?: (?:hd|pro|rx))? \d{3,4}(?!.*m))': '__EXTRACT__'
            },
            'intel': {
                r'intel (?P<graphics_details>arc a\d{3}m)': '__EXTRACT__'
            }
        },
        'integrated': {
            'intel': {
                r'(?P<graphics_details>uhd(?: graphics)?(?: premium| \d{3})?)': '__EXTRACT__',
                r'(?P<graphics_details>hd graphics(?: \d{3,4})?)(?!nvidia)': '__EXTRACT__',
                r'iris plus': 'iris xe plus graphics',
                r'iris(?!.*gpu)|intel xe': 'iris xe graphics',
                r'intel (?P<graphics_details>hd(?: \d{3,4})?)': '__EXTRACT__',
                r'(?P<graphics_details>gt2 graphics)': '__EXTRACT__',
                r'^(?P<graphics_details>hd integrated graphics)$': '__EXTRACT__',
                r'intel (?P<graphics_details>\d{3}u)': '__EXTRACT__'
            },
            'nvidia': {
                r'nvidia geforce gtx (?P<graphics_details>\d{3}m)': '__EXTRACT__'
            },
            'amd': {
                r'amd (?P<graphics_details>radeon (?:rx )?vega (?:3|8|9|11))': '__EXTRACT__',
                r'^(?:integrated )?(?:amd )?radeon(?: graphics)?$': 'radeon graphics',
                r'(?P<graphics_details>radeon (?:(?:rx|hd) )?\d{3,4}m)': '__EXTRACT__',
                r'^amd (?P<graphics_details>integrated graphics)$': '__EXTRACT__',
                r'^amd (?P<graphics_details>radeon graphics 5)$': '__EXTRACT__'
            },
            'mediatek': {
                r'mediatek': np.NaN
            },
            'imagination technologies': {
                r'^(?P<graphics_details>powervr gx6250)$': '__EXTRACT__'
            },
            'apple': {
                r'apple integrated graphics': np.NaN
            },
            'qualcomm': {
                r'(?P<graphics_details>adreno 618)': '__EXTRACT__'
            },
            'arm': {
                r'(?P<graphics_details>mali-g[57]2 (?:mp3|2ee mc2)?)': '__EXTRACT__'
            },
            np.NaN: {
                r'^embedded$|^integrated(?: ?graphics)?$|^shared$': np.NaN
            }
        },
        '__KEEP__': {
            'nvidia': {
                r'nvidia': np.NaN
            },
            'intel': {
                r'intel': np.NaN
            }
        }
    }
    for graphics_type, graphics_brands in graphicsPatterns.items():
        for graphics_brand, patterns in graphics_brands.items():
            for pattern, overwriteValue in patterns.items():
                if match := re.search(pattern, str(row['graphics_coprocessor'])):
                    if graphics_type != '__KEEP__':
                        row['graphics'] = graphics_type
                    row['graphics_brand'] = graphics_brand

                    if (pd.isna(row['graphics_details']) or row['graphics_details'] == ''):
                        if details := match.groupdict().get('graphics_details'):
                            row['graphics_details'] = details
                        else:
                            row['graphics_details'] = overwriteValue
                    
    row['graphics_coprocessor'] = np.NaN
    return row

def standardiseGPU(df):
    # Fix spellings and duplication
    columns = ['graphics', 'graphics_coprocessor']
    replacements = {r'integreted|intergrated': 'integrated', 
                    r'integrated, dedicated': 'dedicated',
                    r'geforcer': 'geforce',
                    r'grpahics|\bgraphic\b': 'graphics',
                    r'\bx\b': 'xe',
                    r'rtx(?! )': 'rtx ',
                    r'\bgt\b': 'gtx'}
    df[columns] = df[columns].replace(replacements, regex=True)

    # Reasoning: Remove unnecessary characters from the column
    df[columns] = df[columns].replace('\s+', ' ', regex=True)
    for column in columns:
        df[column] = df[column].str.strip(' ')

    # Move gpu info to the other column
    def moveGPUinfo(row):
        if pd.notna(row['graphics']):
            if row['graphics'] not in ['integrated', 'dedicated']:
                row['graphics_coprocessor'] = row['graphics']
                row['graphics'] = np.NaN
        return row
    df = df.apply(moveGPUinfo, axis=1)
    
    # Remove all non-alphanumeric characters
    df['graphics_coprocessor'] = df['graphics_coprocessor'].replace(r'[^A-Za-z0-9, \-]', '', regex=True)

    df = df.apply(extractGPUdetails, axis=1)
    df = df.drop(columns=['graphics_coprocessor'])

    return df

def renameColumns(df):
    columnNameHashMap = {
        'screen_size': 'screen_size_inches',
        'color': 'colour',
        'harddisk': 'harddisk_gb',
        'ram': 'ram_gb',
        'cpu_speed': 'cpu_speed_ghz',
        'price': 'price_usd',
        'OS': 'os'
    }
    return df.rename(columns=columnNameHashMap)

def outputDataToFile(df, path):
    df.to_excel(path, index=False)  # index=False to exclude the index column in the output file

def printNumEmpty(df):
    emptyStrings = df.eq('').sum()
    nulls = df.isnull().sum()

    print('COLUMN_NAME'.ljust(21) + '\'\''.rjust(3) + 'null'.rjust(7))

    for column in df.columns.tolist():
        print(column.ljust(21), end="")
        print(str(emptyStrings[column]).rjust(3), end="")
        print(str(nulls[column]).rjust(7), end="\n")

if __name__ == "__main__":
    function_calls = [
        initialCleaning,
        addColumns,
        manuallyCleanModels, 
        standardiseModels, 
        standardiseBrands, 
        standardiseBrandModels,
        manuallyCleanColors, 
        standardiseColors, 
        cleanScreensPrices, 
        standardiseRAMHarddisk, 
        standardiseOS, 
        manuallyCleanCPUs,
        standardiseCPU, 
        standardiseSpecialFeatures, 
        standardiseCPUSpeeds,
        manuallyCleanGPUs,
        standardiseGPU,
        renameColumns,
    ]
    
    df = pd.read_excel('amazon_laptop_2023.xlsx')

    total_elapsed_time = 0

    for function in function_calls:
        start_time = time.time()  # Start the timer
        df = function(df)
        end_time = time.time()  # End the timer
        elapsed_time = (end_time - start_time) * 1000  # Calculate the elapsed time in milliseconds
        total_elapsed_time += elapsed_time
        print(f"Function {function.__name__}".ljust(36) + f"took {elapsed_time:.0f}".ljust(9) + " ms")

    print("".ljust(36) + f"took {total_elapsed_time:.0f}".ljust(9) + " ms")
    outputDataToFile(df, OUT_FILEPATH)

    ### TESTING ###

    print("Total rows: ", df.shape[0])
    print("".join(['-'] * 70))
    print(df.dtypes)
    print("".join(['-'] * 70))
    printNumEmpty(df)
