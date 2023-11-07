1. Drop rows where all values are NaN
2. Convert all rows to lowercase
3. Remove duplicate rows
4. Remove all laptops that don't include any model numbers

5. Extract information from model column
    - Remove ''laptop' from model
    - Extract RAM and DISK
    - Extract 'detachable 2 in 1' and '2 in 1' (they are different)
    - 'victus by hp 15.6 inch gaming laptop pc 15-fa1010nr'
    - Extract color

6. Extract rgb backlit|touchscreen|evo i7-1260p from colours
7. Convert colours to standard colours

8. Convert all 'enovo' to 'lenovo' and other regex
9. Extract toughbook and latitude from brand into model column

10. Remove 'inches' from screen sizes and convert screen sizes to float

11. Adjust ram harddisk sizes to be more consistent
12. ? Convert all ram and harddisk sizes to specific enum values

13. Convert os systems to standard systems

20. Explode all rows where laptops can have multiple colours