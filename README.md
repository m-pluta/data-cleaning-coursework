1. Drop columns that are all NaN
2. Drop rows where all values are NaN
3. Convert all rows to lowercase
4. Remove duplicate rows
5. Remove all laptops that don't include any model numbers

6. Extract information from model column
    - Remove 'laptop' from model
    - Remove 'newest' from model
    - Extract RAM and DISK
    - Extract 'detachable 2 in 1' and '2 in 1' (they are different)
    - 'victus by hp 15.6 inch gaming laptop pc 15-fa1010nr'
    - Extract color

7. Extract rgb backlit|touchscreen|evo i7-1260p from colours

    ? 8. Explode all rows where laptops can have multiple colours
    Considered but not actually correct
    
    9. Convert colours to standard colours

10. Convert all 'enovo' to 'lenovo' and other regex
11. Extract toughbook and latitude from brand into model column

12. Remove redundant brands from model column

12. Remove 'inches' from screen sizes and convert screen sizes to float

13. Adjust ram harddisk sizes to be more consistent
14. ? Convert all ram and harddisk sizes to specific enum values

15. Convert os systems to standard systems

16. Extract information from cpu column such as cpu speed
    - Split CPU column into manufacturer, series, and model




19. Standardise prices to not have dollars
20. Rename columns to more appropriate names
21. If there are identical laptops with different prices, only keep the cheapest