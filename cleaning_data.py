import pandas as pd

# 1. Load data
df = pd.read_csv('Battery_RUL.csv')

# 2. Reconstruct Battery_ID (Crucial for knowing where one battery ends and another begins)
# A new battery starts when the Cycle_Index drops compared to the previous row.
df['Battery_ID'] = (df['Cycle_Index'] < df['Cycle_Index'].shift(1, fill_value=0)).cumsum()

# 3. Filter out the "horrible" rows (Negative times or astronomically large times)
time_cols = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)', 
             'Time constant current (s)', 'Charging time (s)']

valid_mask = True
for col in time_cols:
    valid_mask &= (df[col] >= 0) & (df[col] < 100000)

cleaned_df = df[valid_mask].copy()

# 4. Re-sequence Cycle_Index and RUL per battery
# Sort by Battery_ID and the old Cycle_Index to maintain chronological order
cleaned_df = cleaned_df.sort_values(by=['Battery_ID', 'Cycle_Index'])

# Create new sequential Cycle_Index starting from 1 for each battery
cleaned_df['Cycle_Index'] = cleaned_df.groupby('Battery_ID').cumcount() + 1

# Recalculate RUL: Max Cycle_Index for that battery minus current Cycle_Index
cleaned_df['RUL'] = cleaned_df.groupby('Battery_ID')['Cycle_Index'].transform('max') - cleaned_df['Cycle_Index']

# 5. Clean up and save
cleaned_df = cleaned_df.drop(columns=['Battery_ID'])
cleaned_df.to_csv('Battery_RUL_Cleaned.csv', index=False)

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")