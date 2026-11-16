"""
Convert Markdown document to a professional Microsoft Word DOCX file using Python standard library.
Usage:
    python tools/convert_md_to_docx.py [input.md] [output.docx]
"""
import sys
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import html

def escape_xml(text):
    return html.escape(text, quote=False)

def create_docx(md_path, docx_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.splitlines()

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="260" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="44"/>
      <w:szCs w:val="44"/>
      <w:color w:val="1F497D"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:i/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
      <w:color w:val="595959"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="360" w:after="120"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="32"/>
      <w:szCs w:val="32"/>
      <w:color w:val="1F497D"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="240" w:after="80"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
      <w:color w:val="2E74B5"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="160" w:after="60"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="23"/>
      <w:szCs w:val="23"/>
      <w:color w:val="2E74B5"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    body_xml_parts = []

    def format_inlines(text):
        tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$)', text)
        runs_xml = []
        for token in tokens:
            if not token:
                continue
            if token.startswith('**') and token.endswith('**') and len(token) >= 4:
                content = token[2:-2]
                runs_xml.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape_xml(content)}</w:t></w:r>')
            elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
                content = token[1:-1]
                runs_xml.append(f'<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">{escape_xml(content)}</w:t></w:r>')
            elif token.startswith('`') and token.endswith('`') and len(token) >= 2:
                content = token[1:-1]
                runs_xml.append(f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/><w:color w:val="C7254E"/></w:rPr><w:t xml:space="preserve">{escape_xml(content)}</w:t></w:r>')
            elif token.startswith('$') and token.endswith('$') and len(token) >= 2:
                content = token[1:-1]
                runs_xml.append(f'<w:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/><w:color w:val="1F497D"/></w:rPr><w:t xml:space="preserve">{escape_xml(content)}</w:t></w:r>')
            else:
                runs_xml.append(f'<w:r><w:t xml:space="preserve">{escape_xml(token)}</w:t></w:r>')
        return "".join(runs_xml)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        raw_strip = line.strip()

        if not raw_strip:
            i += 1
            continue

        if raw_strip in ('---', '***', '___'):
            body_xml_parts.append("""<w:p>
  <w:pPr>
    <w:pBdr>
      <w:bottom w:val="single" w:sz="6" w:space="4" w:color="D3D3D3"/>
    </w:pBdr>
    <w:spacing w:before="120" w:after="160"/>
  </w:pPr>
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('>'):
            quote_text = raw_strip.lstrip('>').strip()
            quote_xml = format_inlines(quote_text)
            body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:pBdr>
      <w:left w:val="single" w:sz="18" w:space="8" w:color="1F497D"/>
    </w:pBdr>
    <w:shd w:val="clear" w:color="auto" w:fill="F9FBFD"/>
    <w:ind w:left="400" w:right="400"/>
    <w:spacing w:before="120" w:after="120"/>
  </w:pPr>
  <w:r><w:rPr><w:i/><w:color w:val="333333"/></w:rPr><w:t xml:space="preserve">  </w:t></w:r>
  {quote_xml}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('```'):
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < n and lines[i].strip().startswith('```'):
                i += 1
            
            body_xml_parts.append("""<w:p>
  <w:pPr>
    <w:spacing w:before="120" w:after="40"/>
    <w:keepNext/>
  </w:pPr>
</w:p>""")
            for cl in code_lines:
                escaped_cl = escape_xml(cl)
                body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:shd w:val="clear" w:color="auto" w:fill="F4F5F7"/>
    <w:spacing w:before="20" w:after="20" w:line="220" w:lineRule="auto"/>
    <w:ind w:left="240" w:right="240"/>
  </w:pPr>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="18"/>
      <w:color w:val="262626"/>
    </w:rPr>
    <w:t xml:space="preserve">{escaped_cl}</w:t>
  </w:r>
</w:p>""")
            body_xml_parts.append("""<w:p><w:pPr><w:spacing w:before="40" w:after="120"/></w:pPr></w:p>""")
            continue

        if raw_strip.startswith('|') and raw_strip.endswith('|'):
            table_lines = []
            while i < n and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            parsed_rows = []
            for tl in table_lines:
                cols = [c.strip() for c in tl.strip('|').split('|')]
                if all(re.match(r'^:?-+:?$', c) for c in cols):
                    continue
                parsed_rows.append(cols)

            if parsed_rows:
                tbl_xml = ['<w:tbl>',
                           """<w:tblPr>
                             <w:tblW w:w="9600" w:type="dxa"/>
                             <w:tblBorders>
                               <w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>
                               <w:left w:val="none"/>
                               <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1F497D"/>
                               <w:right w:val="none"/>
                               <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>
                               <w:insideV w:val="none"/>
                             </w:tblBorders>
                             <w:tblCellMar>
                               <w:top w:w="120" w:type="dxa"/>
                               <w:left w:w="160" w:type="dxa"/>
                               <w:bottom w:w="120" w:type="dxa"/>
                               <w:right w:w="160" w:type="dxa"/>
                             </w:tblCellMar>
                           </w:tblPr>"""]

                for row_idx, row in enumerate(parsed_rows):
                    is_header = (row_idx == 0)
                    tr_xml = ['<w:tr>']
                    if is_header:
                        tr_xml.append('<w:trPr><w:tblHeader/></w:trPr>')
                    for cell in row:
                        shd_val = '<w:shd w:val="clear" w:color="auto" w:fill="1F497D"/>' if is_header else ('<w:shd w:val="clear" w:color="auto" w:fill="F8F9FA"/>' if row_idx % 2 == 1 else '<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>')
                        cell_content_xml = format_inlines(cell)
                        if is_header:
                            cell_p = f"""<w:p>
                              <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                              <w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t xml:space="preserve">{escape_xml(cell)}</w:t></w:r>
                            </w:p>"""
                        else:
                            cell_p = f"""<w:p>
                              <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                              {cell_content_xml}
                            </w:p>"""

                        tc_xml = f"""<w:tc>
                          <w:tcPr>{shd_val}</w:tcPr>
                          {cell_p}
                        </w:tc>"""
                        tr_xml.append(tc_xml)
                    tr_xml.append('</w:tr>')
                    tbl_xml.append("".join(tr_xml))
                tbl_xml.append('</w:tbl>')
                body_xml_parts.append("".join(tbl_xml))
                body_xml_parts.append("""<w:p><w:pPr><w:spacing w:before="80" w:after="160"/></w:pPr></w:p>""")
            continue

        if raw_strip.startswith('# '):
            text = raw_strip[2:].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Title"/></w:pPr>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('## '):
            text = raw_strip[3:].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Subtitle"/></w:pPr>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('### '):
            text = raw_strip[4:].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('#### '):
            text = raw_strip[5:].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Heading2"/></w:pPr>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('##### '):
            text = raw_strip[6:].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr><w:pStyle w:val="Heading3"/></w:pPr>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if re.match(r'^[-*]\s+', raw_strip):
            text = re.sub(r'^[-*]\s+', '', raw_strip)
            bullet_char = "•"
            body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:spacing w:before="40" w:after="60"/>
    <w:ind w:left="480" w:hanging="240"/>
  </w:pPr>
  <w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="1F497D"/></w:rPr><w:t xml:space="preserve">{bullet_char}   </w:t></w:r>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        match_num = re.match(r'^(\d+)\.\s+(.*)', raw_strip)
        if match_num:
            num = match_num.group(1)
            text = match_num.group(2)
            body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:spacing w:before="40" w:after="60"/>
    <w:ind w:left="480" w:hanging="240"/>
  </w:pPr>
  <w:r><w:rPr><w:b/><w:color w:val="1F497D"/></w:rPr><w:t xml:space="preserve">{num}.  </w:t></w:r>
  {format_inlines(text)}
</w:p>""")
            i += 1
            continue

        if raw_strip.startswith('$$') and raw_strip.endswith('$$') and len(raw_strip) > 4:
            math_text = raw_strip[2:-2].strip()
            body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:jc w:val="center"/>
    <w:shd w:val="clear" w:color="auto" w:fill="F8F9FA"/>
    <w:spacing w:before="120" w:after="120"/>
  </w:pPr>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>
      <w:sz w:val="23"/>
      <w:color w:val="1F497D"/>
      <w:i/>
    </w:rPr>
    <w:t xml:space="preserve">{escape_xml(math_text)}</w:t>
  </w:r>
</w:p>""")
            i += 1
            continue

        body_xml_parts.append(f"""<w:p>
  <w:pPr>
    <w:spacing w:before="60" w:after="120"/>
  </w:pPr>
  {format_inlines(raw_strip)}
</w:p>""")
        i += 1

    sect_pr = """<w:sectPr>
  <w:pgSz w:w="11906" w:h="16838"/>
  <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>"""

    body_xml_content = "\n".join(body_xml_parts) + "\n" + sect_pr
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
{body_xml_content}
  </w:body>
</w:document>"""

    os.makedirs(os.path.dirname(os.path.abspath(docx_path)), exist_ok=True)
    with zipfile.ZipFile(docx_path, 'w', zipfile.ZIP_DEFLATED) as docx:
        docx.writestr('[Content_Types].xml', content_types)
        docx.writestr('_rels/.rels', root_rels)
        docx.writestr('word/_rels/document.xml.rels', doc_rels)
        docx.writestr('word/styles.xml', styles_xml)
        docx.writestr('word/document.xml', document_xml)

    print(f"Successfully generated DOCX: {docx_path} ({os.path.getsize(docx_path)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        create_docx(sys.argv[1], sys.argv[2])
    else:
        root = Path(__file__).resolve().parent.parent
        src = root / "TECHNICAL_DOCUMENT.md"
        dst = root / "TECHNICAL_DOCUMENT.docx"
        create_docx(str(src), str(dst))
