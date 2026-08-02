import mammoth
import os
import sys
import re

def sanitize_alt_text(markdown):
    """
    Finds Markdown image tags like ![Alt\nText](path) and converts them to
    single-line tags like ![Alt Text](path).
    """
    # Regex to find multi-line ![alt](src) tags
    # This matches the ![, any characters (including newlines), then ], then (src).
    # The flag re.DOTALL makes . match newlines.
    pattern = r'!\[(.*?)]\((.*?)\)'
    
    def replacer(match):
        alt = match.group(1)
        src = match.group(2)
        # Remove all newlines and multiple spaces from alt text
        clean_alt = re.sub(r'\s+', ' ', alt).strip()
        return f'![{clean_alt}]({src})'
    
    return re.sub(pattern, replacer, markdown, flags=re.DOTALL)

def convert_docx_to_markdown(docx_path, output_md_path, images_dir):
    """
    Converts a .docx file to Markdown and extracts images to a local directory.
    Includes sanitization to prevent multi-line image tags.
    """
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} not found.")
        return

    # Ensure images directory exists
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created images directory: {images_dir}")

    image_counter = 0

    def transform_image(image):
        nonlocal image_counter
        image_counter += 1
        
        # Determine extension
        extension = image.content_type.split("/")[-1]
        image_filename = f"image_{image_counter}.{extension}"
        image_path = os.path.join(images_dir, image_filename)
        
        # Save image to disk
        with image.open() as image_bytes:
            with open(image_path, "wb") as f:
                f.write(image_bytes.read())
        
        # Return the path for the Markdown file
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_md_path))
        return {
            "src": rel_image_path.replace("\\", "/")
        }

    with open(docx_path, "rb") as docx_file:
        # Convert to Markdown
        result = mammoth.convert_to_markdown(
            docx_file,
            convert_image=mammoth.images.inline(transform_image)
        )
        
        markdown = result.value
        messages = result.messages
        
        # SANITIZE MARKDOWN: Fix multi-line alt text
        clean_markdown = sanitize_alt_text(markdown)
        
        # Write markdown to file
        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(clean_markdown)
            
        print(f"Successfully converted {docx_path} to {output_md_path}")
        print(f"Extracted {image_counter} images to {images_dir}")
        print(f"Sanitized image tags for better rendering.")
        
        if messages:
            print("\nConversion Messages:")
            # Filter messages to avoid spamming the console
            warning_messages = [m for m in messages if m.type == 'warning']
            if len(warning_messages) > 10:
                print(f"- Total warnings: {len(warning_messages)}")
            else:
                for message in messages:
                    print(f"- {message}")

if __name__ == "__main__":
    docx_file = "thesis.docx"
    output_md = "thesis.md"
    assets_dir = os.path.join("Documents", "05_Assets", "thesis_images")
    
    convert_docx_to_markdown(docx_file, output_md, assets_dir)
