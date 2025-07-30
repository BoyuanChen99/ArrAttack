import pandas as pd

def main():
    data_name = "harmbench"
    data_file = f"./data/jb_data/{data_name}.csv"
    df = pd.read_csv(data_file, header=None)
    
    # Only keep the first column and rename it to 'goal'
    df = df[[1]]
    df.columns = ['goal']
    # Remove the first row
    df = df.iloc[1:].reset_index(drop=True)
    
    # Add a "target" column with formatted instruction
    df['target'] = df['goal'].apply(lambda x: f"Sure, here is a detailed instruction for how to {x.lower()}")
    
    # Optionally, save or print the modified DataFrame
    print(df.head())
    df.to_csv(f"./data/jb_data/{data_name}_processed.csv", index=False)

if __name__ == "__main__":
    main()
