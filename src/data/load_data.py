#! D:\Python\myvenv\Scripts\python.exe

import bz2
import pandas as pd
import os
import argparse

def load_amazon_reviews(file_path, output_path, num_lines=None):
    """Load and process Amazon reviews dataset"""
    data = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for i, line in enumerate(bz2.open(file_path, "rt", encoding="utf8")):
        if num_lines and i == num_lines:
            break
            
        # label 1 is negative and label 2 is positive
        label = 0 if line[:10] == "__label__1" else 1
        text = line[10:]
        
        localResult = {
            "label": label,
            "text": text
        }
        
        data[i] = localResult
    
    df = pd.DataFrame(data).T
    df = df.reset_index().rename(columns={"index": "Id"})
    df.to_csv(output_path, index=False)
    print(f"Processed {len(df)} reviews and saved to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and process Amazon reviews')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--num-lines', type=int, help='Number of lines to process')
    
    args = parser.parse_args()
    load_amazon_reviews(args.input, args.output, args.num_lines)
