import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'lib')]
from native_foreground import strip_page_background


def test_only_leading_full_white_pages_are_removed():
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
      <path fill="#ffffff" d="M0 0L100 0L100 50L0 50Z"/>
      <g transform="matrix(2 0 0 2 0 0)"><path fill="#fff" d="M0 0L50 0L50 25L0 25Z"/></g>
      <path fill="#ffffff" d="M10 10L20 10L20 20L10 20Z"/>
      <path fill="#ffffff" d="M0 0L100 0L100 50L0 50Z"/>
    </svg>'''
    result, removed = strip_page_background(svg)
    assert removed == 2
    paths = [x for x in ET.fromstring(result).iter() if x.tag.endswith('path')]
    assert len(paths) == 2
    assert paths[0].attrib['d'] == 'M10 10L20 10L20 20L10 20Z'


def test_large_white_triangle_is_not_a_page():
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
      <path fill="#fff" d="M0 0L100 0L0 50Z"/>
    </svg>'''
    result, removed = strip_page_background(svg)
    assert removed == 0
