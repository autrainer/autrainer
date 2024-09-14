import datetime
import os
import sys
from typing import Any

from sphinx.application import Sphinx
import toml


def build_authors(authors):
    return ", ".join([a.split(" <")[0] for a in authors])


pyroject = toml.load("../../pyproject.toml")

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("../extensions"))

project = pyroject["tool"]["poetry"]["name"]
description = pyroject["tool"]["poetry"]["description"]
author = build_authors(pyroject["tool"]["poetry"]["authors"])
copyright = f"{datetime.datetime.now().year}, {author}"
release = pyroject["tool"]["poetry"]["version"]


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxarg.ext",
    "sphinxcontrib.jquery",
    "sphinx_hydra_configurations",
    "sphinx_lexers",
]

master_doc = "index"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = {"theme.css"}
html_js_files = ["external_links.js", "highlight_cli.js"]
autosummary_generate = True
autodoc_member_order = "bysource"
autoclass_content = "init"

html_theme_options = {
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc"],
    "navbar_center": [],
    "logo": {
        "alt_text": f"{project} — {description}",
        "text": project,
        "image_dark": "_static/logo_dark.webp",
        "image_light": "_static/logo_light.webp",
    },
    "pygments_light_style": "catppuccin-latte",
    "pygments_dark_style": "catppuccin-mocha",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/autrainer/autrainer",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"**": ["globaltoc"]}
html_favicon = "_static/favicon.ico"
html_title = f"{project} — {description}"


def set_custom_title(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict,
    doctree: Any,
):
    if pagename == app.config.master_doc:
        context["title"] = f"{project} — {description}"


def setup(app: Sphinx):
    app.connect("html-page-context", set_custom_title)
