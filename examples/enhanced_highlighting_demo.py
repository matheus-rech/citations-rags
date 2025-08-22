#!/usr/bin/env python3
"""
Enhanced PDF Citation Highlighting Demo

This demonstrates the new features:
1. Entity-based color coding
2. Fuzzy rectangle fallback matching
3. Annotation notes with entity types
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pdf.pdf_highlighter import highlight_pdf, PageLocationCitation, ENTITY_COLORS, COLOR_NAMES


def demo_enhanced_citations():
    """Create sample citations with different entity types and colors."""
    
    # Sample citations with entity types and custom colors
    sample_citations = [
        {
            "cited_text": "The primary outcome variable was measured at baseline and follow-up",
            "start_page_number": 3,
            "end_page_number": 3,
            "entity_type": "variable",  # Will be colored yellow
        },
        {
            "cited_text": "Statistical analysis was performed using ANOVA",
            "start_page_number": 4, 
            "end_page_number": 4,
            "entity_type": "method",    # Will be colored green
        },
        {
            "cited_text": "The results showed a significant difference (p < 0.05)",
            "start_page_number": 5,
            "end_page_number": 5, 
            "entity_type": "result",    # Will be colored blue
        },
        {
            "cited_text": "In conclusion, our findings support the hypothesis",
            "start_page_number": 7,
            "end_page_number": 7,
            "entity_type": "conclusion", # Will be colored orange
        },
        {
            "cited_text": "Data collection occurred between January and March 2023",
            "start_page_number": 2,
            "end_page_number": 2,
            "entity_type": "data",      # Will be colored magenta
        },
        {
            "cited_text": "This custom citation will be purple",
            "start_page_number": 1,
            "end_page_number": 1,
            "color": "purple",          # Custom color override
        }
    ]
    
    return sample_citations


def print_color_guide():
    """Print available colors and entity types."""
    print("🎨 Available Colors:")
    for name, rgb in COLOR_NAMES.items():
        print(f"  - {name}: RGB{rgb}")
    
    print("\n🏷️ Default Entity Type Colors:")
    for entity, rgb in ENTITY_COLORS.items():
        if entity != "default":
            print(f"  - {entity}: RGB{rgb}")
    
    print("\n📝 Example Citation JSON Format:")
    example = {
        "cited_text": "Your text here...",
        "start_page_number": 3,
        "end_page_number": 3,
        "entity_type": "variable",  # Optional: variable, method, result, conclusion, data, reference
        "color": "blue"            # Optional: overrides entity_type color
    }
    print(json.dumps(example, indent=2))


def create_demo_json_file():
    """Create a demo JSON file for testing."""
    citations = demo_enhanced_citations()
    
    # Create both formats for demo
    formats = {
        "citations_list.json": citations,
        "citations_wrapped.json": {"citations": citations}
    }
    
    for filename, content in formats.items():
        with open(filename, 'w') as f:
            json.dump(content, f, indent=2)
        print(f"📄 Created demo file: {filename}")
    
    return "citations_list.json"


def main():
    print("🚀 Enhanced PDF Citation Highlighting Demo\n")
    
    print_color_guide()
    print("\n" + "="*60)
    
    demo_file = create_demo_json_file()
    
    print(f"\n✨ Usage Examples:")
    print(f"1. Basic highlighting:")
    print(f"   python -m src.rag_pdf.cli highlight your.pdf output.pdf {demo_file}")
    
    print(f"\n2. Override entity type for all citations:")
    print(f"   python -m src.rag_pdf.cli highlight your.pdf output.pdf {demo_file} --entity-type method")
    
    print(f"\n3. Override color for all citations:")
    print(f"   python -m src.rag_pdf.cli highlight your.pdf output.pdf {demo_file} --color purple")
    
    print(f"\n4. Claude ask+highlight with entity type:")
    print(f"   ANTHROPIC_API_KEY=... python -m src.rag_pdf.cli ask-highlight-claude \\")
    print(f"     your.pdf output.pdf \"What are the main variables?\" --entity-type variable --color yellow")
    
    print(f"\n5. Enable fuzzy rectangle matching:")
    print(f"   python -m src.rag_pdf.cli highlight your.pdf output.pdf {demo_file} --enable-fuzzy")
    
    print(f"\n🎯 Features:")
    print(f"  - 🌈 Automatic color coding by entity type")
    print(f"  - 🎨 Custom color overrides") 
    print(f"  - 📦 Fuzzy rectangle matching when exact search fails")
    print(f"  - 📝 Annotation notes with entity types and text previews")
    print(f"  - 🔧 CLI arguments for easy customization")
    
    print(f"\n📊 Fuzzy Matching:")
    print(f"  - Activates when exact text search finds no matches")
    print(f"  - Searches text blocks for 60% word overlap")
    print(f"  - Draws semi-transparent rectangles around matching blocks")
    print(f"  - Adds '[FUZZY MATCH]' notes for identification")


if __name__ == "__main__":
    main()