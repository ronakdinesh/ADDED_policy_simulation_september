# Policy Extraction Framework

A framework for extracting structured policy information from documents using smart chunking and AI-powered extraction.

## Project Structure

The codebase follows a sequential naming pattern to indicate the processing workflow:

1. **a_preprocessing.py**: Text extraction and initial document processing 
2. **b_smart_chunking.py**: Intelligent document chunking based on policy boundaries
3. **c_policy_extractor.py**: Core policy extraction logic using the OpenAI API
4. **d_extract_cli.py**: Command-line interface for running the extraction pipeline

## Usage

### Command Line Interface

The easiest way to use this framework is through the command-line interface:

```bash
python d_extract_cli.py --input <path> [--output <dir>] [--format csv|excel] [--combine]
```

Arguments:
- `--input`, `-i`: Path to a PDF file or directory containing PDF files
- `--output`, `-o`: Output directory for results (default: Output)
- `--format`, `-f`: Output format - csv or excel (default: excel)
- `--combine`, `-c`: Combine all results into a single output file (when processing multiple files)

### Examples

Process a single PDF file:
```bash
python d_extract_cli.py --input "Document Repo Temp/policy_document.pdf" --output custom_output
```

Process all PDFs in a directory and combine results:
```bash
python d_extract_cli.py --input "Document Repo Temp"
```

## File Organization

All outputs including logs, extraction results, and temporary chunks will be stored in the `Output` directory by default:
- Extracted policies: `Output/[filename]_policies.xlsx` or `Output/[filename]_policies.csv`
- Log files: `Output/policy_extraction.log` and timestamped logs
- Chunks: `Output/chunks/[chunk_files]`

## Dependencies

Required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Configuration

The system uses environment variables for API configuration. Create a `.env` file with the following:

```
Azure_OPENAI_API_KEY=your_api_key
Azure_OPENAI_ENDPOINT=your_endpoint
OPENAI_MODEL_NAME=gpt-4o
```

## Output

The extraction process produces structured data in CSV or Excel format containing:
- Policy metadata (name, issuing year, etc.)
- Content fields (summary, goals, objectives, etc.)
- Classification fields (legislative instrument, sector, etc.)
- Source information (filename, page numbers, etc.)

## Future Improvements

- Specialized extraction templates for different policy types
- Enhanced validation for more complete extractions
- Better context handling for improved accuracy 

# Document Agent Policy Extraction

This project provides a comprehensive policy extraction and analysis pipeline using AI. The pipeline consists of multiple scripts that work together to process, chunk, and analyze policy documents.

## Scripts Overview

### b_smart_chunking_agent.py

This script handles the initial processing of policy documents, chunking them intelligently into smaller pieces, and preparing them for analysis.

#### Usage

```bash
python b_smart_chunking_agent.py --input <input_directory> --output <output_directory>
```

#### Options

- `--input`, `-i`: Directory containing the policy documents to process (PDF, Word, etc.)
- `--output`, `-o`: Directory where chunked policy JSON files will be saved (default: "output")
- `--chunk-size`, `-c`: Maximum chunk size in tokens (default: 8000)
- `--overlap`, `-v`: Overlap between chunks in tokens (default: 200)

#### Example

```bash
python b_smart_chunking_agent.py --input Documents/Policies --output output/DMT
```

### c_policy_agent.py

This script processes the chunked policy JSON files created by b_smart_chunking_agent.py, extracts structured information using Azure OpenAI and Pydantic, and saves the results to an Excel file.

#### Usage

```bash
python c_policy_agent.py --input <input_directory> --output <output_file.xlsx>
```

#### Options

- `--input`, `-i`: Input directory containing policy JSON files (required)
- `--output`, `-o`: Output Excel file path (default: "Output/extracted_policies.xlsx")
- `--concurrency`, `-c`: Maximum number of concurrent API calls (default: 5)
- `--batch`, `-b`: Process all policy folders in the input directory
- `--tracking`, `-t`: Excel file for tracking policies (default: "Output/all_policies.xlsx")
- `--process-all`, `-a`: Process all policies regardless of their processing status
- `--use-tracking`: Explicitly use the tracking file for processing
- `--limit`, `-l`: Limit the number of policies to process from the tracking file

#### Examples

Process all policy JSON files in a directory:
```bash
python c_policy_agent.py --input output/DMT --output Output/extracted_policies.xlsx
```

Process with tracking and limit:
```bash
python c_policy_agent.py --input output --output Output/extracted_policies.xlsx --tracking Output/all_policies.xlsx --limit 10
```

Process all policy folders in batch mode:
```bash
python c_policy_agent.py --input output --output Output/extracted_policies.xlsx --batch
```

## Full Pipeline Example

To process a set of policy documents from start to finish:

1. Prepare your policy documents in a directory (e.g., `Documents/Policies`)
2. Run the chunking agent to break documents into manageable pieces:
   ```bash
   python b_smart_chunking_agent.py --input Documents/Policies --output output/DMT
   ```
3. Process the chunked policy files to extract structured information:
   ```bash
   python c_policy_agent.py --input output/DMT --output Output/extracted_policies.xlsx
   ```
4. Review the results in the generated Excel file

## Requirements

- Python 3.7+
- Azure OpenAI API key (set in .env file)
- Required Python packages:
  - pandas
  - openai
  - pydantic
  - pydantic_ai
  - tiktoken
  - dotenv
  - openpyxl 