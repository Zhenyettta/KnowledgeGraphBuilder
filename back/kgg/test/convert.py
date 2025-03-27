import json
import os


def extract_questions_to_file(input_file, output_file, limit=50):
    """
    Extract questions from the input JSONL file and save them to a separate file.
    Limits the output to the specified number of questions.

    Input format:
    {... "paragraphs": [...], "question": "What is the result of the laws of cause and effect called?"}

    Output format - one question per line:
    What is the result of the laws of cause and effect called?
    Another question...

    Parameters:
    -----------
    input_file : str
        Path to the input JSONL file
    output_file : str
        Path to the output file for questions
    limit : int
        Maximum number of questions to extract (default: 50)
    """

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the input file
        input_data = [json.loads(line.strip()) for line in infile if line.strip()]

        # Limit to the first 'limit' items
        input_data = input_data[:limit]

        # Extract and write questions to the output file
        question_count = 0
        for i, item in enumerate(input_data, 1):
            if "question" in item:
                outfile.write(item["question"] + '\n')
                question_count += 1
            else:
                print(f"Warning: No question found in item {i}")

    print(f"Question extraction complete. Generated {question_count} questions. Output written to {output_file}")


def convert_jsonl_format(input_file, output_file, limit=50):
    """
    Convert JSONL file from the provided format with paragraphs to the target format with text field.
    Limits the output to the specified number of files.

    Input format:
    {"id": "2hop__2791_2785", "paragraphs": [{"idx": 0, "title": "...", "paragraph_text": "..."},...]}

    Output format:
    {"id": 1, "text": "..."}
    {"id": 2, "text": "..."}

    Parameters:
    -----------
    input_file : str
        Path to the input JSONL file
    output_file : str
        Path to the output JSONL file
    limit : int
        Maximum number of files to generate (default: 50)
    """

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the input file
        input_data = [json.loads(line.strip()) for line in infile if line.strip()]

        # Limit to the first 'limit' items
        input_data = input_data[:limit]

        # Process each line and write to output file
        for i, item in enumerate(input_data, 1):
            original_id = item.get('id', f"unknown_{i}")

            # Concatenate all paragraphs into a single text
            paragraphs = item.get('paragraphs', [])
            all_texts = []

            for para in paragraphs:
                # Include the title and text when available
                if 'title' in para and 'paragraph_text' in para:
                    all_texts.append(f"{para['title']}: {para['paragraph_text']}")
                elif 'paragraph_text' in para:
                    all_texts.append(para['paragraph_text'])

            # Join all texts with a space
            combined_text = " ".join(all_texts)

            # Create and write the new json line
            new_item = {
                "id": i,
                "text": combined_text
            }

            outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Generated {min(len(input_data), limit)} items. Output written to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "musique_full_v1.0_test.jsonl"
    questions_output_file = "questions.jsonl"
    texts_output_file = "output.jsonl"
    limit = 100
    # Extract questions from the first 50 samples
    extract_questions_to_file(input_file, questions_output_file, limit)

    # Convert text format for the first 50 samples
    convert_jsonl_format(input_file, texts_output_file, limit)
