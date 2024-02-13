# %% [markdown]
# This assignment uses the dataset ['Daily Shelter & Overnight Service Occupance & Capacity'](https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/) year 2023. \
# To run the code below, download and save the file `daily-shelter-overnight-service-occupancy-capacity-2023.csv` in the same directory as the current notebook.

# %% [markdown]
# ## Metadata Review
# 
# The dataset is a compilation of data gathered by individual organizations that provide shelter or overnight service programs. 
# 
# **Published by** The Toronto Shelter and Support Services (TSSS) division, City of Toronto 
# 
# **Contents** Information about shelter and overnight service programs
# 
# **Frequency of update** Daily
# 
# **Metadata** 
# - column names and definitions
# - various measures of data quality
# - types of file format available for download: csv, json, xml
# - a description of service type and capacity type 
# 
# **Limitations** Unaudited data, data completeness and accuracy depend on each program's records and may not reflect actual situation.
# 
# Access to the data is governed by the [Open Government License - Toronto](https://open.toronto.ca/open-data-license/).

# %% [markdown]
# ## Getting started

# %%
import numpy as np 
import pandas as pd

#%matplotlib inline 
import matplotlib.pyplot as plt
import argparse
import yaml
import logging


# %% [markdown]
# #### 1. Load the data

# %%

parser = argparse.ArgumentParser(description = 'Dataset analysis script')
parser.add_argument('infile', type=str,
                    help='Input filename of dataset')
parser.add_argument('origin_outfile', type=str, 
                    help='Output filename of original dataset')
parser.add_argument('user_config_file', type=str,
                    help='user_config_filename')
parser.add_argument('job_config_file', type=str,
                    help='job_config_filename')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='print verbose logs'
                    )
args = parser.parse_args()

config_paths = [args.user_config_file, args.job_config_file]

config = {}
for path in config_paths:
    try:
        with open(path, 'r') as f:
            this_config = yaml.safe_load(f)
            config.update(this_config)
    except FileNotFoundError as e:
        e.add_note(f'FileNotFoundError: {path} cannot be found. Expect a valid path and filename.')
        raise e

logging_level = logging.DEBUG if args.verbose else logging.INFO

logging.basicConfig(
    level = logging_level,
    handlers=[logging.StreamHandler(), logging.FileHandler('python_a2.log')],
    )


df = pd.read_csv(args.infile)
logging.info(f'Successfully loaded {args.infile}')
assert isinstance(df, pd.DataFrame), 'df should be a DataFrame'

# Save the original dataset using the filename format for assignment 2
df.to_csv(args.origin_outfile)


# %% [markdown]
# #### 2. Profile the DataFrame

# %% [markdown]
# **Column names, dtypes**  As shown below, there are three `dtypes`:  int64, object, and float64. Note that the `dtype` for OCCUPANCY_DATE is object.

# %%
df.info()

# %% [markdown]
# Count of NaN in each column

# %%
df.isna().sum()

# %% [markdown]
# Shape of dataframe

# %%
print(f'Shape of the dataset: {df.shape}')

# %% [markdown]
# #### 3. Summary statistics

# %% [markdown]
# Selecting columns with numerical data to summarize, excluding columns containing ID's

# %%
columns_for_summary = ['SERVICE_USER_COUNT', 'CAPACITY_ACTUAL_BED',
       'CAPACITY_FUNDING_BED', 'OCCUPIED_BEDS', 'UNOCCUPIED_BEDS',
       'UNAVAILABLE_BEDS', 'CAPACITY_ACTUAL_ROOM', 'CAPACITY_FUNDING_ROOM',
       'OCCUPIED_ROOMS', 'UNOCCUPIED_ROOMS', 'UNAVAILABLE_ROOMS',
       'OCCUPANCY_RATE_BEDS', 'OCCUPANCY_RATE_ROOMS']

# %% [markdown]
# **Minimum, maximum, mean and median of numeric columns**

# %%
df_numeric_summary =df[columns_for_summary].describe().T[['min','max','mean','50%']].rename(columns={'50%':'median'})
df_numeric_summary

# %% [markdown]
# Note that for `UNAVAILABLE_BEDS` and `UNOCCUPIED_ROOMS`, -1 is listed as their minimum values.  To understand what this means, we go to the definitions on the website: \
# `UNAVAILABLE_BEDS = CAPACITY_FUNDING_BED - CAPACITY_ACTUAL_BED`, \
# `UNOCCUPIED_ROOMS = CAPACITY_ACTUAL_ROOM - OCCUPIED_ROOMS`.

# %% [markdown]
# **Most common values of text columns** \
# Selecting text columns by identifying those with `dtype` as 'object'

# %%
text_columns = df.select_dtypes(include='object').columns

# %% [markdown]
# For most columns, only one value is identified as most common, and can be found in row 0.  For *OCCUPANCY_DATE*, 5 values are listed as most common, and for *PROGRAM_NAME*, 117 values are listed. 

# %%
df[text_columns].mode()

# %% [markdown]
# **Counts of unique values**

# %%
df[text_columns].nunique()

# %% [markdown]
# #### 4. Renaming columns
# Changing all column names to lowercase

# %%
df = df.rename(columns=str.lower)

# %% [markdown]
# #### 5. Unique values of a single column

# %%
df['organization_name'].unique()

# %% [markdown]
# #### 6. Value counts of a text column

# %%
df['location_city'].value_counts()

# %% [markdown]
# #### 7. Convert data type of *occupancy_date*

# %%
# Convert to datetime dtype and remove the time component
#careful, this might not be datetime

df['occupancy_date'] = pd.to_datetime(df['occupancy_date'])
df['occupancy_date']

# %%
# column 'month' is created from 'occupancy_date' for later use
df['month'] = df['occupancy_date'].dt.month
df['month']

# %% [markdown]
# #### 8. Save data to an excel file

# %%
logging.debug(f'Saving DataFrame df as an excel file.')
df.to_excel('VU_NHI_python_assignment2_proc.xlsx', index=False)


# %% [markdown]
# ## More data wrangling

# %% [markdown]
# #### 1. Create a column from an existing one 
# Create a new column, `occupancy_rate`, which takes values from `occupancy_rate_beds` or `occupancy_rate_rooms` if `capacity_type` is 'Bed Based Capacity' or 'Room Based Capacity', respectively.  

# %%
def get_rate(df):
    if (df['capacity_type'] == 'Room Based Capacity'):
        return df['occupancy_rate_rooms']
    elif (df['capacity_type'] == 'Bed Based Capacity'):
        return df['occupancy_rate_beds']
    
df['occupancy_rate'] = df.apply(get_rate, axis=1)
df['occupancy_rate']


# %% [markdown]
# #### 2. Removing a column from the DataFrame

# %%
logging.info('Dropping "location_province" column from DataFrame df.')
df = df.drop('location_province', axis=1)

# %% [markdown]
# #### 3. Extract a subset of columns and rows to a new DataFrame
# Extracting data from shelters/programs that have bed-based capacity

# %%
bed_based_columns = [
 'organization_name',
 'shelter_group',
 'capacity_type',
 'capacity_actual_bed',
 'capacity_funding_bed',
 'occupied_beds',
 'unoccupied_beds',
 'unavailable_beds',
 'occupancy_rate_beds']

# %% [markdown]
# Using .loc[ ]

# %%
bed_df = df.loc[df['capacity_type']=='Bed Based Capacity',
        bed_based_columns]
bed_df

# %% [markdown]
# Using query()

# %%
bed_df2 = df[bed_based_columns].query('capacity_actual_bed > capacity_funding_bed')
bed_df2

# %% [markdown]
# #### 4. Investigate null values

# %%
room_based_columns = ['organization_name',
 'location_id',
 'location_name',
 'location_address',
 'shelter_group',
 'capacity_type',
 'capacity_actual_bed',
 'capacity_funding_bed',
 'occupied_beds',
 'unoccupied_beds',
 'unavailable_beds',
 'capacity_actual_room',
 'capacity_funding_room',
 'occupied_rooms',
 'unoccupied_rooms',
 'unavailable_rooms',
 'occupancy_rate_beds',
 'occupancy_rate_rooms',
 'occupancy_rate']

# %%
room_df = df[room_based_columns]
room_df

# %%
room_df.isna().sum()

# %% [markdown]
# Investigating missing values in 'location_name' column

# %%
room_df.loc[room_df['location_name'].isnull()]

# %%
room_df.loc[room_df['location_name'].isnull()]['location_id'].value_counts()

# %% [markdown]
# From the cells above, we see that all of the NaN's in 'location_name' are from one location, indicating a systematic pattern in missing data.

# %% [markdown]
# Investigating missing data in 'location_address'

# %%

room_df.loc[room_df['location_address'].isnull()]

# %%
room_df.loc[room_df['location_address'].isnull()]['location_id'].value_counts()

# %% [markdown]
# Similarly, from the cells above, we see missing data in 'location_address' concentrate in 3 locations, so they are not randomly scattered.

# %% [markdown]
# The numbers of missing data for columns 
#  `'capacity_actual_bed', 
#  'capacity_funding_bed',
#  'occupied_beds',
#  'unoccupied_beds',
#  'unavailable_beds',
#  'occupancy_rate_bed'`
# 
# are indentical, suggesting a systematic pattern. It is possible that they are missing because these columns don't apply to their organizations' records, because they have room-based capacity.  Let's see if that's the case. 

# %%
df.loc[df['capacity_type']=='Room Based Capacity',
        bed_based_columns]

# %% [markdown]
# There are 15,457 rows of `capacity_type` that are 'Room Based Capacity'.  Similarly, there are exactly 15,457 missing rows for each of the bed-based columns above. \
# Suppose we are only interested in examining room-based shelters data, then we could drop the bed-based columns to get a smaller dataset, as done below.

# %%
room_df = room_df[room_df.capacity_type != 'Bed Based Capacity'].drop(['capacity_actual_bed','capacity_funding_bed',
                                                             'occupied_beds','unoccupied_beds','unavailable_beds',
                                                             'occupancy_rate_beds'], axis=1).head(5)
room_df

# %% [markdown]
# ## Grouping and aggregating

# %% [markdown]
# Create groups based on 'month' using `groupby()` \
# (see how 'month' was created in Section 'Getting started', task 7: Convert data type)
# 

# %%
month_groups = df.groupby(config['groupby'])

# %%
month_groups[['occupancy_rate_beds', 'occupancy_rate_rooms']].mean()

# %% [markdown]
# Using `agg()`

# %%
month_df = df.groupby(config['groupby']).agg(bed_capacity = ('capacity_actual_bed', 'sum'),
                                   beds_occupied = ('occupied_beds', 'sum'),
                                   beds_unavailable = ('unavailable_beds', 'sum'),
                                   bed_occupancy_rate = ('occupancy_rate_beds', 'mean'),
                                   room_capacity =('capacity_actual_room', 'sum'),
                                   rooms_occupied = ('occupied_rooms', 'sum'),
                                   rooms_unavailable = ('unavailable_rooms', 'sum'),
                                   room_occupancy_rate = ('occupancy_rate_rooms', 'mean'))

month_df

# %% [markdown]
# ## Plot

# %% [markdown]
# From the `month_df` dataframe above, we plot `bed_capacity` and `room_capacity` to show how the capacities changed across the 12 months in 2023.

# %%
plt.style.use(config['plot_style'])

fig,ax = plt.subplots()

# %%
# source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

ax.set_axisbelow(True)
ax.grid(alpha=0.3)   
ax.set(xlim=(0, 13), xticks=np.arange(1, 13))

offset = 0.35
bed = ax.bar(month_df.index, month_df['bed_capacity'], width = offset)
room = ax.bar(month_df.index + offset , month_df['room_capacity'], width=offset)

ax.set_title('Room vs. Bed Capacity in Toronto shelters, 2023')
ax.set_xlabel('Month')
ax.set_ylabel('Counts')

ax.legend([bed, room], ['Bed', 'Room'],
          bbox_to_anchor=(1, 1),  # anchor 'it' to coordinate 1,1
          loc='upper left')   # upper left is the 'it' that we're anchoring above

fig



# %%
# why they are blank white fig when opened?
# plt.savefig('room_bed_capacity.png')
# plt.savefig('room_bed_capacity.pdf')
# plt.savefig('room_bed_capacity.jpg')

fig.savefig('room_bed_capacity.png')
fig.savefig('room_bed_capacity.pdf')
fig.savefig('room_bed_capacity.jpg')
# %%
