#!/usr/bin/env python3
"""Turn posts/*.md into posts/*.html using template.html.

    python3 build.py            # build every post
    python3 build.py posts/first-post.md   # build just one

Requires: python3 -m pip install --user markdown
"""

import datetime
import pathlib
import re
import sys

import markdown

ROOT = pathlib.Path(__file__).parent
POSTS = ROOT / "posts"
TEMPLATE = ROOT / "template.html"

# Used to build the BibTeX entry shown at the foot of each post.
# Set SITE_URL to wherever the site is actually published.
AUTHOR = "Stanley Huang"
SITE_URL = "https://example.com"

# Math is pulled out before Markdown runs and put back after, so Markdown
# never sees the LaTeX. Without this, an `x_i` subscript inside $...$ turns
# into <em> and the equation breaks.
DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
BEGIN_ENV = re.compile(r"\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}", re.DOTALL)

# Markdown renders footnotes as a numbered list at the foot of the page. These
# patterns pull that list apart so each note can be moved into the margin
# beside the sentence that cites it.
FOOTNOTE_BLOCK = re.compile(
    r'<div class="footnote">\s*<hr\s*/?>\s*<ol>(.*?)</ol>\s*</div>', re.DOTALL
)
FOOTNOTE_ITEM = re.compile(r'<li id="fn:(.+?)">(.*?)</li>', re.DOTALL)
FOOTNOTE_BACKREF = re.compile(r'\s*<a class="footnote-backref".*?</a>', re.DOTALL)
FOOTNOTE_REF = re.compile(
    r'<sup id="fnref:(.+?)"><a class="footnote-ref" href="#fn:.+?">(.+?)</a></sup>'
)


def protect_math(text):
    """Replace math spans with inert placeholders. Returns (text, spans)."""
    spans = []

    def stash(match):
        spans.append(match.group(0))
        # Alphanumeric only, so no Markdown syntax can apply to it.
        return "zmathspanz{}z".format(len(spans) - 1)

    for pattern in (DISPLAY_MATH, BEGIN_ENV, INLINE_MATH):
        text = pattern.sub(stash, text)
    return text, spans


def restore_math(html, spans):
    for index, span in enumerate(spans):
        html = html.replace("zmathspanz{}z".format(index), span)
    return html


def parse_date(text, fallback):
    """Accept `2026-08-08` or `August 8, 2026`; fall back to the file's mtime."""
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(text.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return fallback


def bibtex(title, stamp, url):
    """A @misc entry others can paste straight into their .bib file."""
    surname = AUTHOR.split()[-1].lower()
    words = re.findall(r"[A-Za-z]+", title.lower())
    keyword = next((w for w in words if len(w) > 3), "post")
    key = "{}{}{}".format(surname, stamp.year, keyword)

    fields = [
        # Double braces stop BibTeX lowercasing the title's capitalisation.
        ("title", "{{%s}}" % title),
        ("author", "{%s}" % AUTHOR),
        ("year", "{%d}" % stamp.year),
        ("month", "{%s}" % stamp.strftime("%b").lower()),
        ("url", "{%s}" % url),
        ("note", "{Blog post}"),
    ]
    width = max(len(name) for name, _ in fields)
    lines = ["@misc{%s," % key]
    lines += ["  {} = {},".format(name.ljust(width), value) for name, value in fields]
    lines.append("}")
    return "\n".join(lines)


def sidenotes(html):
    """Move footnotes from the bottom of the page into the right margin."""
    block = FOOTNOTE_BLOCK.search(html)
    if not block:
        return html

    notes = {}
    for key, body in FOOTNOTE_ITEM.findall(block.group(1)):
        body = FOOTNOTE_BACKREF.sub("", body).replace("&#160;", " ")
        # A sidenote is an inline <span>, which cannot legally contain <p>.
        body = re.sub(r"</p>\s*<p>", "<br><br>", body)
        notes[key] = re.sub(r"</?p>", "", body).strip()

    html = FOOTNOTE_BLOCK.sub("", html)

    def to_sidenote(match):
        key, number = match.group(1), match.group(2)
        if key not in notes:
            return match.group(0)
        # The checkbox is invisible on wide screens; on narrow ones it lets the
        # marker tap open the note inline, since there is no margin to put it in.
        return (
            '<label for="sn-{key}" class="sidenote-ref">{num}</label>'
            '<input type="checkbox" id="sn-{key}" class="sidenote-toggle">'
            '<span class="sidenote"><span class="sidenote-num">{num}</span>{body}</span>'
        ).format(key=key, num=number, body=notes[key])

    return FOOTNOTE_REF.sub(to_sidenote, html)


def split_front_matter(text):
    """Read an optional leading `---` block of key: value lines."""
    meta = {}
    if text.startswith("---"):
        _, block, text = text.split("---", 2)
        for line in block.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip().lower()] = value.strip()
    return meta, text.lstrip("\n")


def build(md_path):
    raw = md_path.read_text(encoding="utf-8")
    meta, body = split_front_matter(raw)

    # Title: front matter, else the first `# heading` (which is then dropped
    # from the body so it does not render twice).
    title = meta.get("title")
    heading = re.match(r"#\s+(.+)", body)
    if heading:
        if not title:
            title = heading.group(1).strip()
        body = body[heading.end():].lstrip("\n")
    title = title or md_path.stem.replace("-", " ").title()

    mtime = datetime.date.fromtimestamp(md_path.stat().st_mtime)
    stamp = parse_date(meta.get("date", ""), mtime)
    date = meta.get("date") or stamp.strftime("%B %-d, %Y")

    body, spans = protect_math(body)
    html = markdown.markdown(
        body,
        extensions=["extra", "sane_lists", "smarty", "codehilite", "toc"],
        extension_configs={
            # Only highlight fenced blocks that name a language, so prose-ish
            # snippets are not guessed at and mis-coloured.
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            # `[TOC]` anywhere in the post expands to a linked outline of the
            # ## and ### headings. Every heading also gets a ¶ permalink.
            "toc": {"permalink": "¶", "toc_depth": "2-3"},
        },
        output_format="html5",
    )
    html = restore_math(html, spans)
    html = sidenotes(html)

    # Let wide tables scroll instead of blowing out the page width.
    html = html.replace("<table>", '<div class="table-scroll"><table>')
    html = html.replace("</table>", "</table></div>")

    url = meta.get("url") or "{}/posts/{}.html".format(SITE_URL.rstrip("/"), md_path.stem)
    entry = bibtex(title, stamp, url)
    entry = entry.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("{{title}}", title).replace("{{date}}", date)
    page = page.replace("{{bibtex}}", entry)
    page = page.replace("{{content}}", html)

    out_path = md_path.with_suffix(".html")
    out_path.write_text(page, encoding="utf-8")
    print("{} -> {}  ({})".format(md_path.name, out_path.name, title))


def main():
    targets = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(POSTS.glob("*.md"))
    if not targets:
        print("No .md files found in posts/")
        return 1
    for path in targets:
        build(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
