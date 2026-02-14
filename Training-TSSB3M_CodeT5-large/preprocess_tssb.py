"""
TSSB-3M Data Preprocessing Pipeline for CodeT5 Fine-tuning
Based on the thesis: "AI-Assisted Code Generation Tools: A New Frontier in Software Development"
"""

import os
import json
import gzip
import random
import requests
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import ast
import re
from collections import Counter

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class TSSB3MPreprocessor:
    """Preprocessor for TSSB-3M dataset to prepare data for CodeT5 fine-tuning."""
    
    def __init__(self, data_dir: str = "./tssb_data", output_dir: str = "./processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_url = "https://zenodo.org/record/5845439/files/tssb_data_3M.zip?download=1"
        self.pattern_explanations = self._create_pattern_explanations()
        
    def _create_pattern_explanations(self) -> Dict[str, str]:
        """Create mapping of SStuB patterns to natural language explanations."""
        explanations = {
            "CHANGE_STRING_LITERAL": "The string literal needs to be corrected to fix the intended behavior.",
            "CHANGE_IDENTIFIER": "The variable or function name needs to be changed to the correct identifier.",
            "CHANGE_NUMERAL": "The numeric value needs to be corrected to fix the calculation.",
            "CHANGE_OPERATOR": "The operator needs to be changed to fix the logic or calculation.",
            "CHANGE_KEYWORD": "The keyword needs to be corrected to fix the control flow.",
            "ADD_FUNCTION_AROUND_EXPRESSION": "A function call needs to be added around this expression to properly process the value.",
            "ADD_EXPRESSION_TO_FUNCTION": "An expression needs to be added to this function call to complete the operation.",
            "SAME_FUNCTION_LESS_ARGS": "The function call has too many arguments and needs to be simplified.",
            "SAME_FUNCTION_MORE_ARGS": "The function call is missing required arguments.",
            "SAME_FUNCTION_DIFFERENT_ARGS": "The arguments in the function call need to be corrected.",
            "SWAP_ARGUMENTS": "The arguments in this function call need to be swapped to match the expected order.",
            "CHANGE_UNARY_OPERATOR": "The unary operator needs to be corrected.",
            "CHANGE_BINARY_OPERATOR": "The binary operator needs to be changed to perform the correct operation.",
            "CHANGE_COMPARISON_OPERATOR": "The comparison operator needs to be corrected to properly compare values.",
            "CHANGE_ASSIGNMENT_OPERATOR": "The assignment operator needs to be fixed.",
            "ADD_UNARY_OPERATOR": "A unary operator needs to be added to fix the expression.",
            "REMOVE_UNARY_OPERATOR": "The unary operator should be removed as it's incorrect.",
            "ADD_BINARY_OPERATOR": "A binary operator needs to be added to combine the expressions.",
            "REMOVE_BINARY_OPERATOR": "The binary operator should be removed.",
            "SINGLE_STMT": "This single statement needs to be corrected.",
            "MULTI_STMT": "Multiple statements need to be adjusted to fix the logic.",
            "MOVE_EXPRESSION": "This expression needs to be moved to a different location.",
            "ADD_IMPORT": "A missing import statement needs to be added.",
            "REMOVE_IMPORT": "An unnecessary import should be removed.",
            "CHANGE_IMPORT": "The import statement needs to be corrected.",
            "ADD_EXCEPTION_HANDLER": "An exception handler needs to be added to handle potential errors.",
            "CHANGE_EXCEPTION_TYPE": "The exception type needs to be corrected to catch the proper error.",
            "ADD_RETURN": "A return statement needs to be added to provide the expected output.",
            "CHANGE_RETURN_VALUE": "The return value needs to be corrected.",
            "ADD_IF_CONDITION": "An if condition needs to be added to handle the specific case.",
            "CHANGE_IF_CONDITION": "The if condition needs to be corrected to properly check the logic.",
            "ADD_LOOP": "A loop needs to be added to iterate over the collection.",
            "CHANGE_LOOP_CONDITION": "The loop condition needs to be corrected.",
            "ADD_ELSE": "An else clause needs to be added to handle the alternative case.",
            "CHANGE_ELSE": "The else clause needs to be corrected.",
            "ADD_ELIF": "An elif clause needs to be added to handle additional conditions.",
            "CHANGE_ELIF": "The elif clause needs to be corrected.",
            "ADD_ATTRIBUTE_ACCESS": "An attribute access needs to be added.",
            "CHANGE_ATTRIBUTE": "The attribute access needs to be corrected.",
            "ADD_SUBSCRIPT": "A subscript/index access needs to be added.",
            "CHANGE_SUBSCRIPT": "The subscript/index access needs to be corrected.",
            "ADD_APPEND": "An append call needs to be added to add the element to the list.",
            "CHANGE_APPEND": "The append call needs to be corrected.",
            "ADD_GET": "A get call needs to be added to safely access the dictionary key.",
            "CHANGE_GET": "The get call needs to be corrected.",
            "ADD_STR": "A str() cast needs to be added to convert the value to string.",
            "ADD_INT": "An int() cast needs to be added to convert the value to integer.",
            "ADD_FLOAT": "A float() cast needs to be added to convert the value to float.",
            "ADD_LIST_CAST": "A list() cast needs to be added to convert the value to list.",
            "ADD_LEN": "A len() call needs to be added to get the length of the collection.",
            "ADD_RANGE": "A range() call needs to be added to generate the sequence.",
            "ADD_ISINSTANCE": "An isinstance() check needs to be added to verify the type.",
            "ADD_PRINT": "A print statement needs to be added to output the result.",
        }
        return explanations

    def download_dataset(self) -> Path:
        """Download TSSB-3M dataset from Zenodo if not already present."""
        zip_path = self.data_dir / "tssb_data_3M.zip"
        
        if zip_path.exists():
            print(f"Dataset already downloaded at {zip_path}")
            return zip_path
            
        print(f"Downloading TSSB-3M dataset from Zenodo...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(self.dataset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Downloaded to {zip_path}")
        return zip_path

    def extract_dataset(self, zip_path: Path) -> Path:
        """Extract the dataset from zip file."""
        extract_dir = self.data_dir / "extracted"
        
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Dataset already extracted at {extract_dir}")
            return extract_dir
            
        print(f"Extracting dataset...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print(f"Extracted to {extract_dir}")
        return extract_dir

    def find_data_files(self, extract_dir: Path) -> List[Path]:
        """Find all relevant data files in the extracted directory."""
        data_files = []
        
        # Convert to string for os.walk to handle Windows paths properly
        extract_path = str(extract_dir.resolve())
        
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.jsonl.gz'):
                    full_path = Path(root) / file
                    data_files.append(full_path)
                    
        print(f"Found {len(data_files)} data files")
        for f in data_files[:5]:
            print(f"  - {f}")
            
        return data_files

    def load_data(self, file_path: Path, sample_size: int = 100000) -> pd.DataFrame:
        """Load data from file and sample specified number of examples."""
        print(f"Loading data from {file_path}...")
        
        # For TSSB-3M, we need to sample from multiple files
        # Each file is .jsonl.gz format
        all_data = []
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line))
                        if len(all_data) >= sample_size * 2:  # Load extra for filtering
                            break
                    except:
                        continue
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line))
                        if len(all_data) >= sample_size * 2:
                            break
                    except:
                        continue
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} total examples")
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"Sampled {sample_size} examples")
            
        return df.reset_index(drop=True)

    def load_from_multiple_files(self, data_files: List[Path], sample_size: int = 100000) -> pd.DataFrame:
        """Load and combine data from multiple files, then sample."""
        print(f"Loading data from {len(data_files)} files...")
        
        all_data = []
        target_per_file = max(1, sample_size // len(data_files) + 1000)  # Load extra from each
        
        for file_path in tqdm(data_files, desc="Loading files"):
            if not str(file_path).endswith('.jsonl.gz'):
                continue
                
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= target_per_file:
                            break
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                            count += 1
                        except:
                            continue
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} total examples from all files")
        
        # Sample to exact size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"Sampled {sample_size} examples")
            
        return df.reset_index(drop=True)

    def parse_edit_script(self, edit_script: str) -> Tuple[str, str, str]:
        """Parse the edit script to extract buggy code, fixed code, and change description."""
        if not edit_script or pd.isna(edit_script):
            return "", "", ""
            
        lines = edit_script.split('\n')
        buggy_lines = []
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                i += 1
                continue
                
            if line.startswith(' '):
                content = line[1:]
                buggy_lines.append(content)
                fixed_lines.append(content)
            elif line.startswith('-'):
                content = line[1:]
                buggy_lines.append(content)
            elif line.startswith('+'):
                content = line[1:]
                fixed_lines.append(content)
            else:
                buggy_lines.append(line)
                fixed_lines.append(line)
                
            i += 1
        
        buggy_code = '\n'.join(buggy_lines)
        fixed_code = '\n'.join(fixed_lines)
        
        return buggy_code, fixed_code, edit_script

    def generate_explanation(self, pattern: str, buggy_code: str, fixed_code: str) -> str:
        """Generate natural language explanation for the bug fix."""
        base_explanation = self.pattern_explanations.get(
            pattern, 
            f"The code needs to be fixed according to the {pattern} pattern."
        )
        
        explanation = base_explanation
        
        if "CHANGE_IDENTIFIER" in pattern:
            buggy_words = set(buggy_code.split())
            fixed_words = set(fixed_code.split())
            diff = buggy_words.symmetric_difference(fixed_words)
            if len(diff) == 2:
                removed, added = list(diff)[:2]
                explanation += f" Change '{removed}' to '{added}'."
                
        elif "CHANGE_NUMERAL" in pattern:
            buggy_nums = re.findall(r'\b\d+\b', buggy_code)
            fixed_nums = re.findall(r'\b\d+\b', fixed_code)
            if buggy_nums and fixed_nums and buggy_nums != fixed_nums:
                explanation += f" Change {buggy_nums[0]} to {fixed_nums[0]}."
                
        elif "CHANGE_OPERATOR" in pattern:
            explanation += " Check the operator precedence and correctness."
            
        elif "ADD_FUNCTION" in pattern:
            explanation += " Ensure the function properly handles the input."
            
        elif "SAME_FUNCTION" in pattern:
            explanation += " Verify the function arguments match the expected signature."
            
        return explanation

    def clean_code(self, code: str) -> str:
        """Clean and normalize code snippet."""
        if not code:
            return ""
            
        lines = code.split('\n')
        
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
                
        if min_indent > 0 and min_indent != float('inf'):
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
            
        return '\n'.join(lines)

    def validate_code(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        if not code or not code.strip():
            return False
            
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def create_instruction_format(
        self, 
        buggy_code: str, 
        explanation: str, 
        pattern: str,
        fixed_code: str = None
    ) -> Dict[str, str]:
        """Create instruction-following format for CodeT5."""
        instruction_explanation = {
            "instruction": "Explain the bug in this code and provide the fix:",
            "input": buggy_code,
            "output": f"Explanation: {explanation}\n\nFixed code:\n{fixed_code if fixed_code else 'No fix available'}"
        }
        
        return instruction_explanation

    def preprocess_data(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Main preprocessing pipeline."""
        processed_data = []
        error_count = 0
        
        print(f"Processing {len(df)} examples...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Extract fields based on TSSB-3M schema
                buggy_code = row.get('source_code', row.get('before_code', row.get('before', row.get('buggy_code', ''))))
                fixed_code = row.get('target_code', row.get('after_code', row.get('after', row.get('fixed_code', ''))))
                pattern = row.get('sstub_pattern', row.get('pattern', row.get('bug_type', 'UNKNOWN')))
                edit_script = row.get('diff', row.get('edit_script', ''))
                
                # If we have edit script but no direct code, parse it
                if not buggy_code and edit_script:
                    buggy_code, fixed_code, _ = self.parse_edit_script(edit_script)
                
                # Clean the code
                buggy_code = self.clean_code(buggy_code)
                fixed_code = self.clean_code(fixed_code)
                
                # Skip if no valid code
                if not buggy_code or not self.validate_code(buggy_code):
                    continue
                    
                # Generate explanation
                explanation = self.generate_explanation(pattern, buggy_code, fixed_code)
                
                # Create instruction format
                instruction_data = self.create_instruction_format(
                    buggy_code, explanation, pattern, fixed_code
                )
                
                # Add metadata
                instruction_data['metadata'] = {
                    'pattern': pattern,
                    'original_index': idx,
                    'has_fix': bool(fixed_code)
                }
                
                processed_data.append(instruction_data)
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_data)} examples")
        print(f"Errors encountered: {error_count}")
        
        return processed_data

    def split_data(
        self, 
        data: List[Dict[str, str]], 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/validation/test sets."""
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data

    def save_data(
        self, 
        data: List[Dict[str, str]], 
        filename: str,
        format: str = "jsonl"
    ):
        """Save processed data to file."""
        output_path = self.output_dir / filename
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        print(f"Saved {len(data)} examples to {output_path}")

    def run_full_pipeline(self, sample_size: int = 100000):
        """Run the complete preprocessing pipeline."""
        print("=" * 60)
        print("TSSB-3M Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Download dataset
        zip_path = self.download_dataset()
        
        # Step 2: Extract dataset
        extract_dir = self.extract_dataset(zip_path)
        
        # Step 3: Find data files
        data_files = self.find_data_files(extract_dir)
        
        if not data_files:
            print("No data files found!")
            raise FileNotFoundError("Could not find data files in extracted archive")
        
        # Step 4: Load and sample data from multiple files
        df = self.load_from_multiple_files(data_files, sample_size=sample_size)
        
        if df is None or len(df) == 0:
            raise ValueError("Could not load any valid data from the dataset")
        
        # Step 5: Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Step 6: Split data
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # Step 7: Save data
        self.save_data(train_data, "train.jsonl")
        self.save_data(val_data, "val.jsonl")
        self.save_data(test_data, "test.jsonl")
        self.save_data(train_data[:100], "train_sample.jsonl")
        
        # Step 8: Generate statistics
        self.generate_statistics(processed_data, train_data, val_data, test_data)
        
        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Output files saved to: {self.output_dir}")
        print(f"  - train.jsonl: {len(train_data)} examples")
        print(f"  - val.jsonl: {len(val_data)} examples")
        print(f"  - test.jsonl: {len(test_data)} examples")
        print(f"  - train_sample.jsonl: 100 examples (for quick testing)")

    def generate_statistics(
        self, 
        full_data: List[Dict], 
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ):
        """Generate and save statistics about the dataset."""
        stats = {
            "total_examples": len(full_data),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "test_examples": len(test_data),
            "pattern_distribution": Counter(),
            "avg_input_length": 0,
            "avg_output_length": 0,
        }
        
        for item in full_data:
            pattern = item.get('metadata', {}).get('pattern', 'UNKNOWN')
            stats['pattern_distribution'][pattern] += 1
            
        total_input_len = sum(len(item['input']) for item in full_data)
        total_output_len = sum(len(item['output']) for item in full_data)
        
        if full_data:
            stats['avg_input_length'] = total_input_len / len(full_data)
            stats['avg_output_length'] = total_output_len / len(full_data)
        
        stats['pattern_distribution'] = dict(stats['pattern_distribution'])
        
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            
        print(f"\nDataset Statistics:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Train/Val/Test split: {stats['train_examples']}/{stats['val_examples']}/{stats['test_examples']}")
        print(f"  Average input length: {stats['avg_input_length']:.1f} chars")
        print(f"  Average output length: {stats['avg_output_length']:.1f} chars")
        print(f"  Number of unique patterns: {len(stats['pattern_distribution'])}")
        print(f"\nTop 10 patterns:")
        for pattern, count in Counter(stats['pattern_distribution']).most_common(10):
            print(f"    {pattern}: {count}")


def main():
    """Main entry point for the preprocessing pipeline."""
    preprocessor = TSSB3MPreprocessor(
        data_dir="./tssb_data",
        output_dir="./processed_data"
    )
    
    preprocessor.run_full_pipeline(sample_size=100000)
    
    print("\nPreprocessing complete! You can now use the processed data for fine-tuning CodeT5.")


if __name__ == "__main__":
    main()