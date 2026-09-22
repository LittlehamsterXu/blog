"""Check a Hugo output tree without requesting external websites (Python 3)."""
import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET


# 交互组件必须在同一页面上带出自己的脚本。
# 只检查“引用的文件在不在”发现不了这类问题：图还在、引用却整个丢了不算断链。
FEATURE_ASSETS = {
    'matrix-demo': ('js/matrix-access.js',),
    # pipeline-demo.js 再 import js/pipeline-model.mjs，那一层由下面的模块 import 检查覆盖。
    'pipeline-demo': ('js/pipeline-demo.js',),
}


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.refs, self.ids, self.remote_code, self.classes = [], set(), [], set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get('id'):
            self.ids.add(a['id'])
        if a.get('class'):
            self.classes.update(a['class'].split())
        for key in ('href', 'src', 'data-zoom-src', 'poster'):
            if a.get(key):
                self.refs.append(a[key])
        if a.get('srcset') and not a['srcset'].startswith('data:'):
            self.refs.extend(item.strip().split()[0] for item in a['srcset'].split(',') if item.strip())
        if tag == 'meta' and a.get('property', a.get('name')) in ('og:image', 'og:url', 'twitter:image'):
            self.refs.append(a.get('content', ''))
        if tag == 'script' or (tag == 'link' and 'stylesheet' in a.get('rel', '')):
            value = a.get('src', a.get('href', ''))
            if value.startswith(('https://', 'http://', '//')):
                self.remote_code.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--base-url', required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    base = args.base_url.rstrip('/') + '/'
    origin = urlsplit(base)
    prefix = unquote(origin.path)
    pages = {p: Document(p.read_text()) for p in root.rglob('*.html')}
    errors = []

    def check(ref, source, label):
        if not ref or ref.startswith(('data:', 'mailto:', 'tel:', 'javascript:')):
            return
        dest = urlsplit(urljoin(source, ref))
        if dest.netloc != origin.netloc or dest.scheme not in ('http', 'https'):
            return
        path = unquote(dest.path)
        if not path.startswith(prefix):
            errors.append(f'{label}: escapes site base path: {ref}')
            return
        target = (root / path[len(prefix):]).resolve()
        if not target.is_relative_to(root):
            errors.append(f'{label}: escapes output tree: {ref}')
            return
        if target.is_dir():
            target /= 'index.html'
        if not target.is_file():
            errors.append(f'{label}: missing target: {ref}')
        elif dest.fragment and target in pages and unquote(dest.fragment) not in pages[target].ids:
            errors.append(f'{label}: missing anchor: {ref}')

    for file, page in pages.items():
        rel = file.relative_to(root).as_posix()
        source = urljoin(base, rel.removesuffix('index.html'))
        for ref in page.refs:
            check(ref, source, rel)
        for ref in page.remote_code:
            if urlsplit(ref).netloc != origin.netloc:
                errors.append(f'{rel}: externally hosted script/style: {ref}')
        referenced = {unquote(urlsplit(ref).path) for ref in page.refs}
        for marker, assets in FEATURE_ASSETS.items():
            if marker not in page.classes:
                continue
            for asset in assets:
                if not any(path.endswith('/' + asset) for path in referenced):
                    errors.append(f'{rel}: .{marker} is present but never loads {asset}')

    for file in root.rglob('*.css'):
        rel = file.relative_to(root).as_posix()
        for ref in re.findall(r'url\(\s*[\'"]?([^\)\'"\s]+)', file.read_text()):
            check(ref, urljoin(base, rel), rel)

    # .js 也要查：入口模块可能就叫 .js，而它 import 的 .mjs 才是真正需要存在的东西。
    modules = sorted(set(root.rglob('*.mjs')) | set(root.rglob('*.js')))
    for file in modules:
        rel = file.relative_to(root).as_posix()
        for ref in re.findall(r'(?:from\s*|import\s*\(\s*|import\s*)[\'"](\.{1,2}/[^\'"]+)[\'"]', file.read_text()):
            check(ref, urljoin(base, rel), rel)

    manifest = root / 'site.webmanifest'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for ref in [data.get('start_url', './')] + [x['src'] for x in data.get('icons', [])]:
            check(ref, urljoin(base, 'site.webmanifest'), 'site.webmanifest')

    for name in ('index.xml', 'sitemap.xml'):
        tree = ET.parse(root / name)
        for node in tree.iter():
            if node.tag.rsplit('}', 1)[-1] in ('loc', 'link', 'url') and node.text and node.text.strip().startswith('http'):
                check(node.text.strip(), base, name)
        for item in tree.findall('./channel/item'):
            content = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
            if content is not None:
                for ref in Document(content.text or '').refs:
                    check(ref, item.findtext('link', base), name)

    for message in sorted(set(errors)):
        print(message, file=sys.stderr)
    if errors:
        return 1
    print(f'OK: {len(pages)} HTML pages, local links/assets, CSS fonts, JS imports, feature assets, RSS and sitemap ({base})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
