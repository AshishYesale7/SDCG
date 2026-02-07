#!/usr/bin/env python3
"""
Convert CGC_EQUATIONS_REFERENCE.txt to PDF
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

def txt_to_pdf(input_file, output_file):
    """Convert text file to PDF with proper formatting."""
    
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    c = canvas.Canvas(str(output_file), pagesize=letter)
    width, height = letter
    
    # Set up font - use Courier for monospace (box-drawing chars)
    c.setFont("Courier", 8)
    
    # Margins
    left_margin = 0.5 * inch
    top_margin = height - 0.5 * inch
    line_height = 9
    
    # Split into lines
    lines = content.split('\n')
    
    y = top_margin
    page_num = 1
    
    for line in lines:
        # Check if we need a new page
        if y < 0.75 * inch:
            # Add page number
            c.setFont("Courier", 8)
            c.drawString(width/2 - 20, 0.4 * inch, f"Page {page_num}")
            c.showPage()
            page_num += 1
            y = top_margin
            c.setFont("Courier", 8)
        
        # Replace unicode box-drawing characters with ASCII equivalents
        line = line.replace('═', '=')
        line = line.replace('║', '|')
        line = line.replace('╔', '+')
        line = line.replace('╗', '+')
        line = line.replace('╚', '+')
        line = line.replace('╝', '+')
        line = line.replace('┌', '+')
        line = line.replace('┐', '+')
        line = line.replace('└', '+')
        line = line.replace('┘', '+')
        line = line.replace('├', '+')
        line = line.replace('┤', '+')
        line = line.replace('┬', '+')
        line = line.replace('┴', '+')
        line = line.replace('┼', '+')
        line = line.replace('─', '-')
        line = line.replace('│', '|')
        line = line.replace('★', '*')
        line = line.replace('✓', '[OK]')
        line = line.replace('✅', '[OK]')
        line = line.replace('❌', '[X]')
        line = line.replace('⚠️', '[!]')
        line = line.replace('📝', '[NOTE]')
        line = line.replace('•', '*')
        line = line.replace('→', '->')
        line = line.replace('←', '<-')
        line = line.replace('↔', '<->')
        line = line.replace('≈', '~')
        line = line.replace('≤', '<=')
        line = line.replace('≥', '>=')
        line = line.replace('×', 'x')
        line = line.replace('÷', '/')
        line = line.replace('±', '+/-')
        line = line.replace('σ', 'sigma')
        line = line.replace('μ', 'mu')
        line = line.replace('β', 'beta')
        line = line.replace('α', 'alpha')
        line = line.replace('γ', 'gamma')
        line = line.replace('δ', 'delta')
        line = line.replace('Δ', 'Delta')
        line = line.replace('Ω', 'Omega')
        line = line.replace('Λ', 'Lambda')
        line = line.replace('ρ', 'rho')
        line = line.replace('π', 'pi')
        line = line.replace('∝', ' prop ')
        line = line.replace('∞', 'inf')
        line = line.replace('ℓ', 'l')
        line = line.replace('⎛', '(')
        line = line.replace('⎝', '(')
        line = line.replace('⎞', ')')
        line = line.replace('⎠', ')')
        
        # Truncate long lines
        max_chars = 115
        if len(line) > max_chars:
            line = line[:max_chars] + '...'
        
        # Draw the line
        try:
            c.drawString(left_margin, y, line)
        except:
            # If there are still problematic characters, skip them
            clean_line = ''.join(c if ord(c) < 128 else '?' for c in line)
            c.drawString(left_margin, y, clean_line)
        
        y -= line_height
    
    # Add final page number
    c.setFont("Courier", 8)
    c.drawString(width/2 - 20, 0.4 * inch, f"Page {page_num}")
    
    c.save()
    print(f"PDF created: {output_file}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    input_file = script_dir / 'CGC_EQUATIONS_REFERENCE.txt'
    output_file = script_dir / 'CGC_EQUATIONS_REFERENCE.pdf'
    
    txt_to_pdf(input_file, output_file)
