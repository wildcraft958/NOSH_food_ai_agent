#!/usr/bin/env python3
"""
Convert REPORT.md to PDF with proper styling.
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import os

def convert_markdown_to_pdf(input_file, output_file):
    """
    Convert a markdown file to PDF with styling.
    
    Args:
        input_file (str): Path to the input markdown file
        output_file (str): Path to the output PDF file
    """
    
    # Read the markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'codehilite', 'fenced_code'])
    html_content = md.convert(markdown_content)
    
    # CSS styling for the PDF
    css_style = """
    @page {
        size: A4;
        margin: 2cm;
        @top-center {
            content: "Food AI Ingredient Scaling Report";
            font-family: Arial, sans-serif;
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: counter(page);
            font-family: Arial, sans-serif;
            font-size: 10pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
        margin: 0;
        padding: 0;
    }
    
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-size: 28px;
    }
    
    h2 {
        color: #34495e;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-size: 22px;
    }
    
    h3 {
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    h4 {
        color: #34495e;
        margin-top: 15px;
        margin-bottom: 8px;
        font-size: 16px;
    }
    
    p {
        text-align: justify;
        margin-bottom: 12px;
    }
    
    ul, ol {
        margin-bottom: 15px;
        padding-left: 25px;
    }
    
    li {
        margin-bottom: 5px;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 14px;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px 8px;
        text-align: left;
    }
    
    th {
        background-color: #f8f9fa;
        font-weight: bold;
        color: #2c3e50;
    }
    
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    tr:hover {
        background-color: #e8f4f8;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 90%;
        color: #e74c3c;
    }
    
    pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        overflow-x: auto;
        margin: 15px 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        color: #333;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 15px 0;
        padding: 10px 20px;
        background-color: #f8f9fa;
        font-style: italic;
    }
    
    .page-break {
        page-break-before: always;
    }
    
    strong {
        color: #2c3e50;
    }
    
    em {
        color: #7f8c8d;
    }
    
    /* Table styling for better readability */
    table strong {
        color: #27ae60;
    }
    
    /* Code highlighting */
    .codehilite {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
    }
    
    /* Image styling to prevent overflow */
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
        box-sizing: border-box;
    }
    
    /* Figure styling */
    figure {
        margin: 20px 0;
        text-align: center;
    }
    
    figcaption {
        font-style: italic;
        color: #7f8c8d;
        margin-top: 10px;
        font-size: 14px;
    }
    """
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Food AI Ingredient Scaling Report</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    html_doc = HTML(string=full_html, base_url=str(Path(input_file).parent))
    css_doc = CSS(string=css_style)
    
    html_doc.write_pdf(output_file, stylesheets=[css_doc])
    print(f"PDF successfully created: {output_file}")

def main():
    """Main function to convert REPORT.md to PDF."""
    
    # Define file paths
    input_file = "REPORT.md"
    output_file = "output/REPORT.pdf"
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        return
    
    # Convert to PDF
    try:
        convert_markdown_to_pdf(input_file, output_file)
        print(f"Report successfully converted to PDF!")
        print(f"Location: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error converting to PDF: {e}")

if __name__ == "__main__":
    main()
