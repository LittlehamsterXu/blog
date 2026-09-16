"""Regression checks for deployment path and missing-resource detection."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parents[1] / 'scripts/check-site.py'


class SiteCheck(unittest.TestCase):
    def check_page(self, html, extras=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {'index.html': html, 'index.xml': '<rss><channel/></rss>',
                     'sitemap.xml': '<urlset/>', **(extras or {})}
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            return subprocess.run([sys.executable, str(CHECKER), directory,
                                   '--base-url', 'https://example.org/blog/'],
                                  capture_output=True, text=True)

    def test_valid_encoded_anchor_and_external_link(self):
        result = self.check_page('<h1 id="标题">Title</h1><a href="#%E6%A0%87%E9%A2%98">Heading</a>'
                                 '<a href="https://other.example/page/">Other site</a>')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_link_cannot_escape_repository_path(self):
        result = self.check_page('<a href="/archives/">Archives</a>')
        self.assertIn('escapes site base path', result.stderr)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_image_and_anchor(self):
        result = self.check_page('<img src="missing.png"><a href="#missing">Link</a>')
        self.assertIn('missing target', result.stderr)
        self.assertIn('missing anchor', result.stderr)
        self.assertNotEqual(result.returncode, 0)

    def test_css_font_and_module_import_are_checked(self):
        result = self.check_page('<link href="style.css">', {
            'style.css': '@font-face {src:url("font.woff2")}',
            'app.mjs': 'import "./missing.mjs";'})
        self.assertIn('font.woff2', result.stderr)
        self.assertIn('missing.mjs', result.stderr)
        self.assertNotEqual(result.returncode, 0)

    def test_cdn_code_is_rejected(self):
        result = self.check_page('<script src="https://cdn.example/library.js"></script>')
        self.assertIn('externally hosted script/style', result.stderr)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
