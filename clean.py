import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\dgons\Desktop\visualisation\data\train.csv')

# Define a dictionary to map month names to their corresponding numerical values
month_dict = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}

# Convert 'Month' column to numerical values based on the dictionary
df['Month'] = df['Month'].map(month_dict)

columns_to_process = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                      'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Age', 
                      'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                      'Credit_Utilization_Ratio', 'Outstanding_Debt']

for col in columns_to_process:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('_', ''), errors='coerce')

# Sort the DataFrame by 'Customer_ID' and 'Month'
df = df.sort_values(by=['Customer_ID', 'Month'])

print('1')

# Iterate over the rows
for index, row in df.iterrows():
    current_customer_id = row['Customer_ID']

    # Fill in missing Monthly_Inhand_Salary with Annual_Income/12
    if pd.isnull(row['Monthly_Inhand_Salary']) and not pd.isnull(row['Annual_Income']):
        df.at[index, 'Monthly_Inhand_Salary'] = row['Annual_Income'] / 12

    # Check if it's not the first row for a customer
    if index > 0 and current_customer_id == df.at[index - 1, 'Customer_ID']:
        # Columns to compare and update
        columns_to_check = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                             'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan']

        # Compare values with the previous row for specified columns
        for col in columns_to_check:
            if row[col] != df.at[index - 1, col]:
                df.at[index, col] = df.at[index - 1, col]

        check_corrupt = ['Name', 'SSN']

        # Compare values with the next row for specified columns
        for col in check_corrupt:
            if pd.isnull(row[col]) or row[col] == "#F%$D@*&8":
                next_value = df.at[index + 1, col]
                if not pd.isnull(next_value) and next_value != "#F%$D@*&8":
                    df.at[index, col] = next_value

    age_value = row['Age']
    if not (0 <= age_value <= 122):
        # Look for a valid 'Age' value from the same 'Customer_ID'
        valid_age = df[(df['Customer_ID'] == current_customer_id) & (0 <= df['Age']) & (df['Age'] <= 120)]['Age'].iloc[0]
        df.at[index, 'Age'] = valid_age

print("done 1")
print("1")

for index, row in df.iterrows():
    current_customer_id = row['Customer_ID']

    if index == 99999:
        break

    if index > 0 and current_customer_id == df.at[index + 1, 'Customer_ID']:
        # Columns to compare and update
        columns_to_check = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                             'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan']

        # Compare values with the previous row for specified columns
        for col in columns_to_check:
            if row[col] != df.at[index + 1, col]:
                df.at[index, col] = df.at[index + 1, col]

        check_corrupt = ['Name', 'SSN']

        # Compare values with the next row for specified columns
        for col in check_corrupt:
            if pd.isnull(row[col]) or row[col] == "#F%$D@*&8":
                next_value = df.at[index + 1, col]
                if not pd.isnull(next_value) and next_value != "#F%$D@*&8":
                    df.at[index, col] = next_value
    
    age_value = row['Age']
    if not (0 <= age_value <= 122):
        # Look for a valid 'Age' value from the same 'Customer_ID'
        valid_age = df[(df['Customer_ID'] == current_customer_id) & (0 <= df['Age']) & (df['Age'] <= 120)]['Age'].iloc[0]
        df.at[index, 'Age'] = valid_age

print("done 2")



# Save the updated DataFrame to a new CSV file
df.to_csv(r'C:\Users\dgons\Desktop\visualisation\cleaned.csv', index=False)
