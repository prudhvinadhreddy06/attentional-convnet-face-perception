import pandas as pd

# Read the CSV file
df = pd.read_csv('baseline (1).csv')

# Round the 'age' column values
df['age'] = df['age'].round().astype(int)

# Write the modified data to a new CSV file
df.to_csv('rounded_age.csv', index=False)

print("Rounded age values saved to 'rounded_age.csv'.")
