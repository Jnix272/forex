import os
import re


def check_markdown_links(directory):
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath) as f:
                    content = f.read()

                for match in link_pattern.finditer(content):
                    link_text = match.group(1)
                    link_target = match.group(2)

                    if (
                        link_target.startswith("http")
                        or link_target.startswith("#")
                        or link_target.startswith("mailto:")
                    ):
                        continue

                    # Remove anchor tags from target file path
                    target_file = link_target.split("#")[0]
                    if not target_file:
                        continue

                    # Calculate absolute path assuming relative links
                    if os.path.isabs(target_file):
                        target_path = target_file
                    else:
                        target_path = os.path.normpath(os.path.join(root, target_file))

                    if not os.path.exists(target_path):
                        print(f"{filepath}: Broken link '{link_text}' pointing to '{link_target}'")


check_markdown_links("docs")
